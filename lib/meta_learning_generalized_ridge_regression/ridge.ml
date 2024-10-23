open Torch
open Types
open Matrix_ops

(* Generalized ridge regression estimator *)
let estimate ~x ~y ~a ~n_l =
  let xt = Tensor.transpose x ~dim0:0 ~dim1:1 in
  let xtx = Tensor.mm xt x in
  let xty = Tensor.mm xt y in
  let n = Tensor.of_float (float_of_int n_l) in
  let reg_term = Tensor.mul_scalar n (Tensor.inverse a) in
  let term = Tensor.add xtx reg_term in
  let inv_term = Tensor.inverse term in
  Tensor.mm inv_term xty

(* Oracle predictive risk *)
let oracle_risk ~x_new ~y_new ~beta ~omega =
  let pred = Tensor.mm x_new beta in
  let diff = Tensor.sub pred y_new in
  let squared_error = Tensor.mm (Tensor.transpose diff ~dim0:0 ~dim1:1) diff in
  Tensor.to_float0_exn squared_error

(* Compute asymptotic risk *)
let compute_asymptotic_risk ~x ~y ~omega ~sigma_sq ~gamma =
  let dim = Tensor.size x |> List.nth 1 in
  let lambda = gamma *. sigma_sq in

  let omega_sqrt = matrix_power omega 0.5 in
  let xt = Tensor.transpose x ~dim0:0 ~dim1:1 in
  let scaled_x = Tensor.mm x omega_sqrt in
  let gram = Tensor.mm scaled_x (Tensor.transpose scaled_x ~dim0:0 ~dim1:1) in
  
  let resolvent = Tensor.inverse (Tensor.add gram (Tensor.eye dim |> Tensor.mul_scalar lambda)) in
  
  let term1 = Tensor.mm resolvent gram |> Tensor.trace |> Tensor.to_float0_exn in
  let term1 = term1 /. gamma in
  
  let term2 = 1. -. gamma in
  let term2 = term2 *. (
    sigma_sq +. (lambda *. gamma -. sigma_sq) *. sigma_sq *. gamma *.
    (Tensor.trace resolvent |> Tensor.to_float0_exn) +.
    (1. -. gamma) *. gamma *. term1
  ) /. (gamma *. term1 +. (1. -. gamma)) in
  
  term1 +. term2

(* Compute Riemannian gradient for optimality verification *)
let compute_riemannian_gradient ~omega ~sigma ~gamma ~sigma_sq =
  let dim = Tensor.size omega |> List.hd in
  let inv_omega = Tensor.inverse omega in
  let scaled_sigma = Tensor.mul_scalar sigma (gamma *. sigma_sq /. float_of_int dim) in
  let term1 = Tensor.mm scaled_sigma inv_omega in
  let term2 = Tensor.mm term1 scaled_sigma in
  Tensor.sub term2 (Tensor.eye dim)

(* Compute predictive risk matrix *)
let compute_risk_matrix ~x ~y ~omega ~sigma_sq =
  let dim = Tensor.size x |> List.nth 1 in
  let xt = Tensor.transpose x ~dim0:0 ~dim1:1 in
  let beta = estimate ~x ~y ~a:omega ~n_l:(Tensor.size x |> List.hd) in
  let pred = Tensor.mm x beta in
  let diff = Tensor.sub pred y in
  let risk_matrix = Tensor.mm (Tensor.transpose diff ~dim0:0 ~dim1:1) diff in
  Tensor.add risk_matrix (Tensor.eye dim |> Tensor.mul_scalar sigma_sq)