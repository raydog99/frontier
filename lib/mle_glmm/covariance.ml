open Types
open Torch
open MatrixOps

(* Matérn covariance *)
let matern_covariance params dist_mat =
  let open Tensor in
  let scaled_dist = mul_scalar dist_mat (scalar_float params.range) in
  
  let nu = scalar_float params.smoothness in
  let variance = scalar_float params.variance in
  
  let gamma_term = Scalar.float (
    exp (Stats.log_gamma params.smoothness) *. 
    sqrt (Float.pi *. 2.0 ** (1.0 -. params.smoothness))
  ) in
  
  let k_nu = bessel_k nu scaled_dist in
  
  mul_scalar (mul k_nu 
                 (pow scaled_dist (mul_scalar nu 
                                  (scalar_float params.smoothness))))
             variance |>
  div (scalar_float gamma_term)

(* Compute full covariance matrix *)
let compute_covariance spec dist_mat =
  match spec.covariance with
  | Matern params -> matern_covariance params dist_mat
  | Exponential {variance; range} ->
      matern_covariance 
        {variance; range; smoothness = 0.5}
        dist_mat
  | Independence {variance} ->
      Tensor.eye (Tensor.size dist_mat 0) |>
      Tensor.mul_scalar (Scalar.float variance)

(* Derivative of Matérn covariance with respect to parameters *)
let derivative_matern params dist_mat =
  let open Tensor in
  let scaled_dist = mul_scalar dist_mat (scalar_float params.range) in
  let nu = scalar_float params.smoothness in
  
  let d_variance = div (matern_covariance params dist_mat) 
                      (scalar_float params.variance) in
  
  let d_range = 
    let k_deriv = bessel_k (add nu (scalar_float 1.0)) scaled_dist in
    mul_scalar (mul k_deriv scaled_dist) 
               (scalar_float (-1.0 *. params.smoothness)) in
  
  stack [d_variance; d_range] ~dim:0

(* Derivative of full covariance matrix *)
let derivative_covariance spec dist_mat =
  match spec.covariance with
  | Matern params -> derivative_matern params dist_mat
  | Exponential {variance; range} ->
      derivative_matern 
        {variance; range; smoothness = 0.5}
        dist_mat
  | Independence {variance} ->
      Tensor.ones [1; Tensor.size dist_mat 0; Tensor.size dist_mat 1] |>
      Tensor.mul_scalar (Scalar.float variance)