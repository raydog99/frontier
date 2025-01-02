open Torch
open Types

type optimizer_state = {
  params: model_params;
  grad: Tensor.t;
  momentum: Tensor.t option;
  iteration: int;
}

let adam lr (beta1, beta2) epsilon state =
  let open Tensor in
  
  (* Initialize momentum if needed *)
  let m = match state.momentum with
    | Some m -> m
    | None -> zeros_like state.grad
  in
  
  (* Update biased first moment estimate *)
  let m' = Tensor.(
    add
      (mul (float_tensor [beta1]) m)
      (mul (float_tensor [1.0 -. beta1]) state.grad)
  ) in
  
  (* Update biased second raw moment estimate *)
  let v = Tensor.(mul state.grad state.grad) in
  let v' = Tensor.(
    add
      (mul (float_tensor [beta2]) v)
      (mul (float_tensor [1.0 -. beta2]) v)
  ) in
  
  (* Compute bias-corrected first moment estimate *)
  let m_hat = Tensor.(
    div m'
      (sub (float_tensor [1.0]) 
        (pow (float_tensor [beta1]) (float_tensor [Float.of_int state.iteration])))
  ) in
  
  (* Compute bias-corrected second raw moment estimate *)
  let v_hat = Tensor.(
    div v'
      (sub (float_tensor [1.0])
        (pow (float_tensor [beta2]) (float_tensor [Float.of_int state.iteration])))
  ) in
  
  (* Compute update *)
  let update = Tensor.(
    div
      (mul (float_tensor [lr]) m_hat)
      (add (sqrt v_hat) (float_tensor [epsilon]))
  ) in
  
  (* Apply update *)
  let new_params = {
    beta = Tensor.(sub state.params.beta update);
    u = state.params.u;  (* Random effects not updated by optimizer *)
    lambda = state.params.lambda;
  } in
  
  {state with params = new_params; momentum = Some m'; iteration = state.iteration + 1}

let newton_raphson data params max_iter tol =
  let rec iterate params iter =
    if iter >= max_iter then params
    else
      let grad = Gradients.log_posterior data params params.prior_params in
      let fisher = Fisher_Info.compute data params in
      let update = Tensor.(mm (inverse fisher) grad.beta) in
      
      let new_params = {params with
        beta = Tensor.(sub params.beta update)
      } in
      
      if Tensor.float_value (Tensor.norm update) < tol then
        new_params
      else
        iterate new_params (iter + 1)
  in
  iterate params 0