open Types
open Torch

let create ~input_dim ~expert_type =
  let config = {
    n_experts = 2;
    input_dim;
    expert_type;
    symmetric = true;
  } in
  let expert_param = Base_moe.create_parameter [input_dim; 1] in
  let gate_param = Base_moe.create_parameter [input_dim; 1] in
  {
    experts = expert_param;
    gates = gate_param;
    config;
  }

let compute_gate_probability input gate_params =
  let logits = Tensor.mm input gate_params in
  let exp_term = Tensor.exp (Tensor.mul_scalar logits 0.5) in
  Tensor.div exp_term (Tensor.add (Tensor.ones_like exp_term) exp_term)

let compute_linear_conditional input expert_params target =
  let pred = Tensor.mm input expert_params in
  let diff = Tensor.sub pred target in
  let squared_diff = Tensor.mul diff diff in
  Tensor.div squared_diff (Tensor.ones_like squared_diff |> Tensor.full_like ~value:2.0)
  |> Tensor.neg
  |> Tensor.exp

let compute_logistic_conditional input expert_params target =
  let logits = Tensor.mm input expert_params in
  let pos_prob = Tensor.sigmoid logits in
  let neg_prob = Tensor.sub (Tensor.ones_like pos_prob) pos_prob in
  Tensor.where target pos_prob neg_prob

let forward model input =
  let pos_expert = model.experts in
  let neg_expert = Tensor.neg model.experts in
  let gate_logits = Tensor.mm input model.gates in
  
  let pos_output = 
    Base_moe.expert_forward 
      ~expert_params:pos_expert 
      ~input 
      ~expert_type:model.config.expert_type in
  let neg_output = 
    Base_moe.expert_forward 
      ~expert_params:neg_expert 
      ~input 
      ~expert_type:model.config.expert_type in
  
  let gate_probs = compute_gate_probability input model.gates in
  
  let pos_contrib = Tensor.mul gate_probs pos_output in
  let neg_contrib = 
    Tensor.mul 
      (Tensor.sub (Tensor.ones_like gate_probs) gate_probs) 
      neg_output in
  Tensor.add pos_contrib neg_contrib

let compute_mirror_map model input =
  match model.config.expert_type with
  | Linear ->
      let x_squared = Tensor.mul input input in
      let expert_term = 
        Tensor.sum x_squared ~dim:1
        |> fun t -> Tensor.div t (Tensor.ones [1] ~kind:Float |> Tensor.full_like ~value:2.0) in
      let gate_term = 
        Tensor.mm input model.gates
        |> fun t -> Tensor.log1p (Tensor.exp t) in
      Tensor.mean (Tensor.add expert_term gate_term) |> Tensor.item
  | Logistic ->
      let expert_term = 
        Tensor.mm input model.experts
        |> fun t -> Tensor.log1p (Tensor.exp t) in
      let gate_term =
        Tensor.mm input model.gates
        |> fun t -> Tensor.log1p (Tensor.exp t) in
      Tensor.mean (Tensor.add expert_term gate_term) |> Tensor.item