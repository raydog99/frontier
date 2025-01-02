open Types
open Torch

let create_parameter shape =
  Tensor.randn shape ~kind:Float

let softmax x =
  let max_x = Tensor.max x ~dim:1 ~keepdim:true |> fst in
  let exp_x = Tensor.exp (Tensor.sub x max_x) in
  let sum_exp_x = Tensor.sum exp_x ~dim:1 ~keepdim:true in
  Tensor.div exp_x sum_exp_x

let create ~config =
  let expert_shape = [config.input_dim; config.n_experts] in
  let gate_shape = [config.input_dim; config.n_experts] in
  {
    experts = create_parameter expert_shape;
    gates = create_parameter gate_shape;
    config;
  }

let expert_forward ~expert_params ~input ~expert_type =
  match expert_type with
  | Linear ->
      Tensor.mm input expert_params
  | Logistic ->
      let logits = Tensor.mm input expert_params in
      Tensor.sigmoid logits

let gate_forward ~gate_params ~input =
  let logits = Tensor.mm input gate_params in
  softmax logits

let forward model input =
  let expert_outputs = 
    expert_forward 
      ~expert_params:model.experts
      ~input 
      ~expert_type:model.config.expert_type in
  
  let gate_probs = gate_forward ~gate_params:model.gates ~input in
  
  Tensor.sum (Tensor.mul gate_probs expert_outputs) ~dim:1

let expert_loss ~expert_params ~input ~target ~expert_type =
  match expert_type with
  | Linear ->
      let pred = expert_forward ~expert_params ~input ~expert_type in
      let diff = Tensor.sub pred target in
      Tensor.mean (Tensor.mul diff diff)
  | Logistic ->
      let logits = Tensor.mm input expert_params in
      let bce = Tensor.binary_cross_entropy_with_logits ~target logits in
      Tensor.mean bce

let gate_loss ~gate_params ~input ~expert_outputs ~target ~expert_type =
  let gate_probs = gate_forward ~gate_params ~input in
  let weighted_outputs = Tensor.mul gate_probs expert_outputs in
  match expert_type with
  | Linear ->
      let diff = Tensor.sub weighted_outputs target in
      Tensor.mean (Tensor.mul diff diff)
  | Logistic ->
      let log_probs = Tensor.log weighted_outputs in
      Tensor.mean (Tensor.neg (Tensor.mul target log_probs))

let loss model input target =
  let expert_outputs = 
    expert_forward 
      ~expert_params:model.experts
      ~input
      ~expert_type:model.config.expert_type in
  
  gate_loss 
    ~gate_params:model.gates
    ~input
    ~expert_outputs
    ~target
    ~expert_type:model.config.expert_type

let compute_log_likelihood model input target =
  let expert_outputs = 
    expert_forward 
      ~expert_params:model.experts
      ~input 
      ~expert_type:model.config.expert_type in
  let gate_probs = gate_forward ~gate_params:model.gates ~input in
  
  let joint_probs = Tensor.mul expert_outputs gate_probs in
  let sum_probs = Tensor.sum joint_probs ~dim:1 in
  Tensor.mean (Tensor.log sum_probs)

let train_experts model input target =
  let expert_logits = Tensor.mm input model.experts in
  let expert_outputs = match model.config.expert_type with
    | Linear -> expert_logits
    | Logistic -> Tensor.sigmoid expert_logits in
  
  let grad = Tensor.grad_of_fn 
    (fun p -> expert_loss ~expert_params:p ~input ~target ~expert_type:model.config.expert_type)
    model.experts in
  let updated_experts = Tensor.sub model.experts grad in
  { model with experts = updated_experts }

let train_gates model input target =
  let expert_outputs = expert_forward 
    ~expert_params:model.experts
    ~input 
    ~expert_type:model.config.expert_type in
  
  let grad = Tensor.grad_of_fn
    (fun p -> gate_loss ~gate_params:p ~input ~expert_outputs ~target ~expert_type:model.config.expert_type)
    model.gates in
  let updated_gates = Tensor.sub model.gates grad in
  { model with gates = updated_gates }