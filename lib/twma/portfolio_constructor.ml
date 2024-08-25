open Torch

type method_t = 
  | EqualWeight
  | MeanVariance of float  (* risk_aversion parameter *)
  | RiskParity
  | BlackLitterman of (Tensor.t * Tensor.t)  (* views and confidences *)

type t = {
  method_: method_t;
  num_assets: int;
}

let create method_ num_assets =
  { method_; num_assets }

let equal_weight t =
  Tensor.(ones [t.num_assets] / float_of_int t.num_assets)

let mean_variance t returns covariance risk_aversion =
  let expected_returns = Tensor.mean returns ~dim:[0] in
  let inverse_covariance = Tensor.inverse covariance in
  let temp = Tensor.(matmul inverse_covariance expected_returns) in
  Tensor.(temp / (sum temp))

let risk_parity t covariance =
  let rec newton_raphson x max_iter tol =
    if max_iter = 0 then x else
    let diag_x = Tensor.diag x in
    let f = Tensor.(matmul (matmul covariance diag_x) x - ones [t.num_assets]) in
    let j = Tensor.(matmul covariance diag_x + matmul (matmul covariance x) (transpose x ~dim0:0 ~dim1:1)) in
    let dx = Tensor.(matmul (inverse j) f) in
    let x_new = Tensor.(x - dx) in
    if Tensor.(sum (abs dx)) |> Tensor.to_float0_exn < tol then x_new
    else newton_raphson x_new (max_iter - 1) tol
  in
  let initial_guess = Tensor.(ones [t.num_assets] / float_of_int t.num_assets) in
  let x = newton_raphson initial_guess 100 1e-6 in
  Tensor.(x / (sum x))

let black_litterman t returns covariance (views, confidences) =
  let prior_returns = Tensor.mean returns ~dim:[0] in
  let tau = 0.05 in  (* Typical value, can be adjusted *)
  let omega = Tensor.(diag confidences) in
  let weighted_views = Tensor.(matmul views (inverse omega)) in
  let posterior_covariance = Tensor.(
    inverse (inverse (mul_scalar covariance tau) + matmul (transpose views ~dim0:0 ~dim1:1) (matmul (inverse omega) views))
  ) in
  let posterior_returns = Tensor.(
    matmul posterior_covariance 
      (matmul (inverse (mul_scalar covariance tau)) prior_returns + 
       matmul (transpose views ~dim0:0 ~dim1:1) weighted_views)
  ) in
  mean_variance t posterior_returns posterior_covariance 1.0

let construct_portfolio t returns covariance =
  match t.method_ with
  | EqualWeight -> equal_weight t
  | MeanVariance risk_aversion -> mean_variance t returns covariance risk_aversion
  | RiskParity -> risk_parity t covariance
  | BlackLitterman (views, confidences) -> black_litterman t returns covariance (views, confidences)