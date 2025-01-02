open Torch
open Types

let step data state epsilon =
  let grad = Gradients.log_posterior data state.params state.prior_params in
  
  (* Generate proposal *)
  let noise_beta = Tensor.randn (Tensor.size state.params.beta) in
  let noise_u = Tensor.randn (Tensor.size state.params.u) in
  let noise_lambda = Tensor.randn (Tensor.size state.params.lambda) in
  
  let proposal = {
    beta = Tensor.(add state.params.beta 
      (add 
        (mul grad.beta (float_tensor [epsilon /. 2.0]))
        (mul noise_beta (sqrt (float_tensor [epsilon])))));
    u = Tensor.(add state.params.u
      (add
        (mul grad.u (float_tensor [epsilon /. 2.0]))
        (mul noise_u (sqrt (float_tensor [epsilon])))));
    lambda = Tensor.(add state.params.lambda
      (add
        (mul grad.lambda (float_tensor [epsilon /. 2.0]))
        (mul noise_lambda (sqrt (float_tensor [epsilon])))));
  } in
  
  (* Compute acceptance probability *)
  let forward_ll = Models.Logistic.log_likelihood data proposal in
  let backward_ll = Models.Logistic.log_likelihood data state.params in
  
  let accept_prob = Float.exp (forward_ll -. backward_ll) in
  
  if Random.float 1.0 < accept_prob then
    {state with 
      params = proposal;
      log_prob = forward_ll;
      accepted = state.accepted + 1;
      total = state.total + 1}
  else
    {state with total = state.total + 1}

let manifold_step data state epsilon =
  let g = Fisher_Info.compute data state.params in
  let g_inv = Numerical.safe_inverse g in
  let grad = Gradients.log_posterior data state.params state.prior_params in
  
  (* Transform gradient using metric tensor *)
  let transformed_grad = Tensor.mm g_inv grad.beta in
  let proposal = {state.params with
    beta = Tensor.(add state.params.beta
      (add
        (mul transformed_grad (float_tensor [epsilon /. 2.0]))
        (mul (mm (safe_cholesky g_inv) (randn (size state.params.beta)))
           (sqrt (float_tensor [epsilon])))))
  } in
  
  (* Compute acceptance probability with metric tensor *)
  let forward_ll = Models.Logistic.log_likelihood data proposal in
  let backward_ll = Models.Logistic.log_likelihood data state.params in
  
  let accept_prob = Float.exp (forward_ll -. backward_ll) in
  
  if Random.float 1.0 < accept_prob then
    {state with
      params = proposal;
      log_prob = forward_ll;
      accepted = state.accepted + 1;
      total = state.total + 1}
  else
    {state with total = state.total + 1}