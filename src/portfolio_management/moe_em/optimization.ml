open Types
open Torch

module EM = struct
  let e_step model input target =
    let pos_expert = model.experts in
    let neg_expert = Tensor.neg model.experts in
    
    let pos_output = Base_moe.expert_forward 
      ~expert_params:pos_expert
      ~input
      ~expert_type:model.config.expert_type in
    let neg_output = Base_moe.expert_forward 
      ~expert_params:neg_expert
      ~input
      ~expert_type:model.config.expert_type in
    
    let gate_probs = Base_moe.gate_forward ~gate_params:model.gates ~input in
    
    let pos_joint = Tensor.mul pos_output (Tensor.select gate_probs ~dim:1 ~index:0) in
    let neg_joint = Tensor.mul neg_output (Tensor.select gate_probs ~dim:1 ~index:1) in
    
    let total = Tensor.add pos_joint neg_joint in
    let pos_resp = Tensor.div pos_joint total in
    let neg_resp = Tensor.div neg_joint total in
    
    (pos_resp, neg_resp)

  let m_step model input target (pos_resp, neg_resp) =
    (* Update expert parameters *)
    let weighted_pos = Tensor.mul input pos_resp in
    let weighted_neg = Tensor.mul input neg_resp in
    
    let pos_contrib = Tensor.mm (Tensor.transpose weighted_pos ~dim0:0 ~dim1:1) target in
    let neg_contrib = Tensor.mm (Tensor.transpose weighted_neg ~dim0:0 ~dim1:1) target in
    
    let expert_update = Tensor.div 
      (Tensor.sub pos_contrib neg_contrib)
      (Tensor.ones [1] ~kind:Float |> Tensor.full_like ~value:2.0) in
    
    (* Update gate parameters *)
    let gate_update = Tensor.mm (Tensor.transpose input ~dim0:0 ~dim1:1) pos_resp in
    
    { model with
      experts = expert_update;
      gates = gate_update;
    }

  let train ?(config={max_iterations=100; tolerance=1e-6; relative_smoothness=1.0}) 
            model input target =
    let rec loop model state =
      if state.iteration >= config.max_iterations || state.converged then
        (model, state)
      else
        let (pos_resp, neg_resp) = e_step model input target in
        let updated_model = m_step model input target (pos_resp, neg_resp) in
        
        let current_loss = Base_moe.loss updated_model input target in
        let converged = Float.abs (current_loss -. state.loss) < config.tolerance in
        
        let new_state = {
          iteration = state.iteration + 1;
          loss = current_loss;
          kl_divergence = 0.0;
          converged;
        } in
        
        loop updated_model new_state in
    
    loop model {
      iteration = 0;
      loss = Float.infinity;
      kl_divergence = 0.0;
      converged = false;
    }
end

module MirrorDescent = struct
  let compute_kl_divergence model1 model2 input =
    let p1 = Symmetric_moe.forward model1 input in
    let p2 = Symmetric_moe.forward model2 input in
    
    let kl = Tensor.mul p1 (Tensor.log (Tensor.div p1 p2))
             |> Tensor.mean
             |> Tensor.item in
    Float.max kl 0.0

  let compute_gradients model input target =
    let input = Tensor.set_requires_grad input true in
    
    (* Compute forward pass and loss *)
    let output = Symmetric_moe.forward model input in
    let loss = match model.config.expert_type with
      | Linear -> 
          let diff = Tensor.sub output target in
          Tensor.mean (Tensor.mul diff diff)
      | Logistic ->
          Tensor.binary_cross_entropy_with_logits ~target output in
    
    (* Compute gradients *)
    let grad_output = Tensor.ones [1] ~kind:Float in
    Tensor.backward loss [grad_output];
    
    let expert_grad = Tensor.grad model.experts in
    let gate_grad = Tensor.grad model.gates in
    
    (expert_grad, gate_grad)

  let compute_regularized_loss model input target =
    let mirror_map = Symmetric_moe.compute_mirror_map model input in
    let loss = Base_moe.compute_log_likelihood model input target in
    Tensor.add_scalar loss mirror_map

  let train ?(config={max_iterations=100; tolerance=1e-6; relative_smoothness=1.0}) 
            model input target =
    let rec loop model state =
      if state.iteration >= config.max_iterations || state.converged then
        (model, state)
      else
        let expert_grad, gate_grad = compute_gradients model input target in
        
        (* Update parameters using mirror descent *)
        let updated_experts =
          Tensor.sub model.experts 
            (Tensor.mul_scalar expert_grad config.relative_smoothness) in
        let updated_gates =
          Tensor.sub model.gates
            (Tensor.mul_scalar gate_grad config.relative_smoothness) in
        
        let updated_model = { model with experts = updated_experts; gates = updated_gates } in
        
        (* Compute metrics *)
        let current_loss = Base_moe.loss updated_model input target in
        let kl_div = compute_kl_divergence model updated_model input in
        let converged = Float.abs (current_loss -. state.loss) < config.tolerance in
        
        let new_state = {
          iteration = state.iteration + 1;
          loss = current_loss;
          kl_divergence = kl_div;
          converged;
        } in
        
        loop updated_model new_state in
    
    loop model {
      iteration = 0;
      loss = Float.infinity;
      kl_divergence = 0.0;
      converged = false;
    }
end