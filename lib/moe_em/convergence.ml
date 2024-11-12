open Types
open Torch

module FisherInfo = struct
  let compute_complete_data_fisher model input target =
    let batch_size = Tensor.size input 0 in
    let param_dim = Tensor.size model.experts 0 in
    
    (* Enable gradient computation *)
    let input = Tensor.set_requires_grad input true in
    let output = Symmetric_moe.forward model input in
    
    let loss = match model.config.expert_type with
      | Linear -> Tensor.mse_loss output target
      | Logistic -> Tensor.binary_cross_entropy_with_logits ~target output in
    
    let grad_output = Tensor.ones [1] ~kind:Float in
    Tensor.backward loss [grad_output];
    
    let expert_grad = Tensor.grad model.experts |> Tensor.reshape ~shape:[param_dim; -1] in
    let gate_grad = Tensor.grad model.gates |> Tensor.reshape ~shape:[param_dim; -1] in
    
    let fisher = Tensor.cat [expert_grad; gate_grad] ~dim:1 in
    Tensor.mm (Tensor.transpose fisher ~dim0:0 ~dim1:1) fisher
    |> Tensor.div_scalar (Float.of_int batch_size)

  let compute_conditional_fisher model input target =
    let batch_size = Tensor.size input 0 in
    let param_dim = Tensor.size model.experts 0 in
    
    let pos_resp, neg_resp = EM.e_step model input target in
    let resp_grad = 
      Tensor.sub pos_resp neg_resp
      |> Tensor.log
      |> Tensor.set_requires_grad ~requires_grad:true in
    
    let grad_output = Tensor.ones [1] ~kind:Float in
    Tensor.backward resp_grad [grad_output];
    
    let expert_grad = Tensor.grad model.experts |> Tensor.reshape ~shape:[param_dim; -1] in
    let gate_grad = Tensor.grad model.gates |> Tensor.reshape ~shape:[param_dim; -1] in
    
    let fisher = Tensor.cat [expert_grad; gate_grad] ~dim:1 in
    Tensor.mm (Tensor.transpose fisher ~dim0:0 ~dim1:1) fisher
    |> Tensor.div_scalar (Float.of_int batch_size)

  let compute_mim model input target =
    let complete_fisher = compute_complete_data_fisher model input target in
    let conditional_fisher = compute_conditional_fisher model input target in
    
    let complete_inv = Tensor.inverse complete_fisher in
    Tensor.mm complete_inv conditional_fisher
end

module Analysis = struct
  let compute_eigenvalues tensor =
    let eigenvalues, _ = Tensor.symeig tensor ~eigenvectors:true in
    eigenvalues

  let estimate_snr model input target =
    let param_norm = 
      Tensor.cat [model.experts; model.gates] ~dim:1
      |> Tensor.norm ~p:2 ~dim:[0; 1]
      |> Tensor.item in
    let output = Symmetric_moe.forward model input in
    let residuals = Tensor.sub output target in
    let noise = Tensor.std residuals |> Tensor.item in
    param_norm /. noise

  let check_conditions model input target =
    let mim = FisherInfo.compute_mim model input target in
    let max_eigenvalue = 
      compute_eigenvalues mim
      |> Tensor.max ~dim:0 ~keepdim:false
      |> fst
      |> Tensor.item in
    
    let snr = estimate_snr model input target in
    
    {
      locally_convex = max_eigenvalue < 1.0;
      relatively_strongly_convex = max_eigenvalue < 0.5;
      sufficient_snr = snr > 4.0;
    }

  let compute_metrics model input target =
    let mim = FisherInfo.compute_mim model input target in
    let max_eigenvalue = 
      compute_eigenvalues mim
      |> Tensor.max ~dim:0 ~keepdim:false
      |> fst
      |> Tensor.item in
    
    let relative_convexity = 1.0 -. max_eigenvalue in
    let snr = estimate_snr model input target in
    
    {
      mim_max_eigenvalue = max_eigenvalue;
      relative_convexity;
      snr_estimate = snr;
      convergence_rate = 
        if max_eigenvalue < 1.0 then Some (1.0 -. relative_convexity)
        else None;
    }

  let verify_relative_smoothness model input target =
    let loss_hessian = 
      let loss = Base_moe.compute_log_likelihood model input target in
      Tensor.hessian loss (Tensor.cat [model.experts; model.gates] ~dim:1) in
    
    let mirror_hessian =
      let mirror_map = Symmetric_moe.compute_mirror_map model input in
      Tensor.hessian mirror_map (Tensor.cat [model.experts; model.gates] ~dim:1) in
    
    let relative_smooth_matrix = Tensor.sub loss_hessian mirror_hessian in
    let eigenvalues = compute_eigenvalues relative_smooth_matrix in
    
    Tensor.all (Tensor.le eigenvalues (Tensor.zeros_like eigenvalues))
    |> Tensor.item
    |> Bool.to_float

  let analyze_snr model input target =
    let param_dim = Tensor.size model.experts 0 in
    let gate_norm = Tensor.norm model.gates ~p:2 ~dim:[0; 1] |> Tensor.item in
    let expert_norm = Tensor.norm model.experts ~p:2 ~dim:[0; 1] |> Tensor.item in
    
    let complete_fisher_scaling = match model.config.expert_type with
      | Linear ->
          let w_scaling = Float.pow gate_norm 1.5 in
          let w_squared = gate_norm *. gate_norm in
          [|w_scaling; w_squared; 1.0|]
      | Logistic ->
          let w_scaling = Float.pow gate_norm 1.5 in
          let w_squared = gate_norm *. gate_norm in
          let phi_scaling = Float.pow expert_norm 1.5 in
          let phi_squared = expert_norm *. expert_norm in
          [|w_scaling; w_squared; phi_scaling; phi_squared|] in
    
    let conditional_scaling = 
      1.0 /. (Float.pow (gate_norm *. gate_norm +. expert_norm *. expert_norm) 1.5) in
    
    let mim_eigenvalue_bound = 
      Array.fold_left (fun acc scale -> max acc (scale *. conditional_scaling))
        0.0 complete_fisher_scaling in
    
    (mim_eigenvalue_bound, complete_fisher_scaling, conditional_scaling)
end