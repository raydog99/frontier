open Torch

type optimizer_state = {
  parameters: Tensor.t;
  velocity: Tensor.t;
  momentum: float;
  learning_rate: float;
}

let riemannian_sgd geometry state grad =
  let metric = geometry.InformationGeometry.metric.metric_tensor in
  let riem_grad = 
    Tensor.matmul 
      (Tensor.inverse (metric state.parameters)) 
      grad in
  
  let velocity =
    Tensor.add
      (Tensor.mul_scalar state.velocity state.momentum)
      (Tensor.mul_scalar riem_grad 
         (1. -. state.momentum)) in
  
  let update =
    Tensor.mul_scalar velocity state.learning_rate in
  
  let new_params =
    InformationGeometry.exponential_map
      geometry
      state.parameters
      update in
  
  {state with
    parameters = new_params;
    velocity = velocity}

let riemannian_adam geometry state grad =
  let beta1 = 0.9 in
  let beta2 = 0.999 in
  let epsilon = 1e-8 in
  
  let metric = geometry.InformationGeometry.metric.metric_tensor in
  let riem_grad = 
    Tensor.matmul 
      (Tensor.inverse (metric state.parameters)) 
      grad in
  
  (* Update biased first moment estimate *)
  let m = 
    Tensor.add
      (Tensor.mul_scalar state.velocity beta1)
      (Tensor.mul_scalar riem_grad (1. -. beta1)) in
  
  (* Update biased second raw moment estimate *)
  let v =
    Tensor.add
      (Tensor.mul_scalar 
         (Tensor.mul riem_grad riem_grad) 
         (1. -. beta2))
      (Tensor.mul_scalar state.velocity beta2) in
  
  (* Compute bias-corrected first moment estimate *)
  let m_hat =
    Tensor.div_scalar m (1. -. beta1) in
  
  (* Compute bias-corrected second raw moment estimate *)
  let v_hat =
    Tensor.div_scalar v (1. -. beta2) in
  
  (* Compute update *)
  let update =
    Tensor.div
      (Tensor.mul_scalar m_hat state.learning_rate)
      (Tensor.add
         (Tensor.sqrt v_hat)
         (Tensor.scalar epsilon)) in
  
  let new_params =
    InformationGeometry.exponential_map
      geometry
      state.parameters
      update in
  
  {state with
    parameters = new_params;
    velocity = m}