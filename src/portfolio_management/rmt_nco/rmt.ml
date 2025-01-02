open Torch

let tracy_widom_test eigenvalue n p alpha =
  let mu_np = (Float.sqrt n -. 1. +. Float.sqrt p) ** 2. in
  let sigma_np = (Float.sqrt n -. 1. +. Float.sqrt p) *. (1. /. Float.sqrt n -. 1. +. 1. /. Float.sqrt p) ** (1. /. 3.) in
  let s = (eigenvalue -. mu_np) /. sigma_np in
  (* Approximation of Tracy-Widom CDF *)
  let f_beta s = exp (-. (1. /. 24.) *. s ** 3. -. (1. /. 8.) *. s ** 2.) in
  f_beta s > 1. -. alpha

let estimate_tracy_widom data alpha =
  let n, p = Tensor.shape2_exn data in
  let cov = Tensor.mm (Tensor.transpose2 data) data in
  let eigenvalues, eigenvectors = Tensor.symeig ~eigenvectors:true cov in
  let filtered_eigenvalues = 
    Tensor.map (fun e -> if tracy_widom_test e n p alpha then e else Tensor.mean eigenvalues) eigenvalues
  in
  Tensor.mm (Tensor.mm eigenvectors (Tensor.diag filtered_eigenvalues)) (Tensor.transpose2 eigenvectors)

let estimate_linear_shrinkage data =
  let n, p = Tensor.shape2_exn data in
  let cov = Tensor.mm (Tensor.transpose2 data) data in
  let f_norm_cov = Tensor.frobenius_norm cov in
  let f_norm_cov_minus_id = Tensor.frobenius_norm (Tensor.sub cov (Tensor.eye p)) in
  let alpha = min ((1. /. Float.of_int n) *. f_norm_cov) (f_norm_cov_minus_id ** 2.) /. (f_norm_cov_minus_id ** 2.) in
  Tensor.add (Tensor.mul_scalar cov (1. -. alpha)) (Tensor.mul_scalar (Tensor.eye p) alpha)

let estimate_covariance data method_ =
  match method_ with
  | `Tracy_Widom -> estimate_tracy_widom data 0.05  (* Using 5% significance level *)
  | `Linear_Shrinkage -> estimate_linear_shrinkage data
  | `Naive -> Tensor.mm (Tensor.transpose2 data) data