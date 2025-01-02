open Torch

let refine_sqrt_epsilon ~current ~samples ~epsilon =
  let d = Tensor.size samples 1 in
  let current_inv_sqrt = Numerical_stability.stable_inverse_sqrt current in
  let rotated = Fast_matrix_ops.rect_multiply current_inv_sqrt samples in
  
  let z_samples = Batch_processor.efficient_kronecker 
    ~samples:rotated ~batch_size:1000 in
  let mean_est = Robust_mean_estimator.estimate_bounded_covariance 
    ~samples:z_samples ~epsilon in
  
  reconstruct_covariance current mean_est

let refine_log_epsilon ~current ~samples ~epsilon ~tau =
  let d = Tensor.size samples 1 in
  let current_inv_sqrt = Numerical_stability.stable_inverse_sqrt current in
  let rotated = Fast_matrix_ops.rect_multiply current_inv_sqrt samples in
  
  let z_samples = Batch_processor.efficient_kronecker 
    ~samples:rotated ~batch_size:1000 in
  let mean_est = Robust_mean_estimator.estimate_known_covariance 
    ~samples:z_samples ~epsilon ~tau in
  
  reconstruct_covariance current mean_est

let verify_refinement ~estimate ~previous ~epsilon =
  let diff = Tensor.sub estimate previous in
  let rel_change = Tensor.frobenius_norm diff |>
    Tensor.float_value in
  let prev_norm = Tensor.frobenius_norm previous |>
    Tensor.float_value in
  rel_change <= epsilon *. prev_norm

let reconstruct_covariance current mean_est =
  let d = Tensor.size current 0 in
  let mean_matrix = Tensor.reshape mean_est [|d; d|] in
  let current_sqrt = Numerical_stability.stable_matrix_sqrt current in
  
  Numerical_stability.stable_matrix_multiply
    (Numerical_stability.stable_matrix_multiply current_sqrt mean_matrix)
    current_sqrt