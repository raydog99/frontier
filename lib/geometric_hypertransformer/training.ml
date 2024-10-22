open Torch

type optimizer_config = {
  learning_rate: float;
  momentum: float;
  weight_decay: float;
}

type training_state = {
  model: (Tensor.t, Tensor.t) GeometricHypertransformer.t;
  optimizer: optimizer_config;
  iteration: int;
  best_loss: float;
}

let create_training_state ~model optimizer_config = {
  model;
  optimizer = optimizer_config;
  iteration = 0;
  best_loss = Float.infinity;
}

(* Riemannian optimization step *)
let riemannian_step state grad =
  let module Geom = InformationGeometry in
  let manifold = state.model.base_transformer.geometry in
  
  (* Compute Riemannian gradient *)
  let metric = manifold.metric.metric_tensor state.model.memory.parameters in
  let riem_grad = Tensor.matmul (Tensor.inverse metric) grad in
  
  (* Apply momentum using parallel transport *)
  let transported_velocity = 
    if state.iteration > 0 then
      Geom.parallel_transport manifold
        state.model.memory.parameters
        (Tensor.add 
           state.model.memory.parameters
           (Tensor.mul_scalar riem_grad 
              state.optimizer.learning_rate))
        state.model.memory.velocity
    else
      Tensor.zeros_like riem_grad in
  
  let velocity =
    Tensor.add
      (Tensor.mul_scalar transported_velocity 
         state.optimizer.momentum)
      (Tensor.mul_scalar riem_grad 
         (1. -. state.optimizer.momentum)) in
  
  (* Update parameters along geodesic *)
  let update_vector =
    Tensor.add
      (Tensor.mul_scalar velocity state.optimizer.learning_rate)
      (Tensor.mul_scalar state.model.memory.parameters 
         (-. state.optimizer.weight_decay)) in
  
  let new_params =
    Geom.exponential_map manifold
      state.model.memory.parameters
      update_vector in
  
  {state.model.memory with
    parameters = new_params;
    velocity = velocity}

let train_epoch state training_data =
  let total_loss = ref 0. in
  let num_batches = List.length training_data in
  
  let final_state = List.fold_left (fun state (input, target) ->
    (* Forward pass *)
    let output = 
      GeometricHypertransformer.forward 
        state.model input in
    
    (* Compute loss *)
    let module QAS = (val state.model.qas_space) in
    let loss = QAS.distance output target in
    total_loss := !total_loss +. loss;
    
    (* Backward pass *)
    let grad = Tensor.grad loss in
    
    (* Update parameters *)
    let new_memory = riemannian_step state grad in
    
    {state with
      model = {state.model with memory = new_memory};
      iteration = state.iteration + 1;
      best_loss = min state.best_loss loss}
  ) state training_data in
  
  final_state, !total_loss /. float_of_int num_batches

let evaluate state validation_data =
  let total_loss = ref 0. in
  let num_batches = List.length validation_data in
  
  List.iter (fun (input, target) ->
    let output = 
      GeometricHypertransformer.forward 
        state.model input in
    
    let module QAS = (val state.model.qas_space) in
    let loss = QAS.distance output target in
    total_loss := !total_loss +. loss
  ) validation_data;
  
  !total_loss /. float_of_int num_batches