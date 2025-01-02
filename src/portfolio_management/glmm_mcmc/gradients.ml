open Torch
open Types

let log_likelihood data params =
  let eta = Glmm_base.linear_predictor data.x data.z params.beta params.u in
  let grad_beta = Tensor.(mm (transpose data.x) 
    (sub data.y (Numerical.stable_sigmoid eta))) in
  let grad_u = Tensor.(mm (transpose data.z)
    (sub data.y (Numerical.stable_sigmoid eta))) in
  let grad_lambda = Tensor.(
    div
      (sub 
        (mul params.lambda params.lambda)
        (mul params.u params.u))
      (mul (float_tensor [2.0]) (mul params.lambda params.lambda))
  ) in
  {beta = grad_beta; u = grad_u; lambda = grad_lambda}

let prior params prior_params =
  let grad_beta = Tensor.(
    div
      (sub params.beta prior_params.mu_0)
      (mul (float_tensor [2.0]) prior_params.q)
  ) in
  let grad_u = Tensor.(
    div params.u (mul (float_tensor [2.0]) params.lambda)
  ) in
  let grad_lambda = Tensor.(
    sub
      (div (float_tensor [1.0]) params.lambda)
      (div
        (mul params.u params.u)
        (mul (float_tensor [2.0]) (mul params.lambda params.lambda)))
  ) in
  {beta = grad_beta; u = grad_u; lambda = grad_lambda}

let log_posterior data params prior_params =
  let ll_grad = log_likelihood data params in
  let prior_grad = prior params prior_params in
  {
    beta = Tensor.(add ll_grad.beta prior_grad.beta);
    u = Tensor.(add ll_grad.u prior_grad.u);
    lambda = Tensor.(add ll_grad.lambda prior_grad.lambda)
  }