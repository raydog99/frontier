open Torch

type t = {
  mean_vector: Tensor.t;
  covariance_matrix: Tensor.t;
  n: int;
  p: int;
}

let create mean_vector covariance_matrix =
  let n = Tensor.size mean_vector 0 in
  let p = Tensor.size covariance_matrix 0 in
  { mean_vector; covariance_matrix; n; p }

let estimate_parameters = Estimators.estimate

let consistent_estimators ef =
  let c = float_of_int ef.p /. float_of_int ef.n in
  let (r_gmv, v_gmv, s) = estimate_parameters ef Estimators.Consistent in
  let r_c = r_gmv in
  let v_c = Tensor.(v_gmv / (1.0 - c)) in
  let s_c = Tensor.(s * (1.0 - c)) in
  (r_c, v_c, s_c)

let asymptotic_normality ef =
  let c = float_of_int ef.p /. float_of_int ef.n in
  let (r_c, v_c, s_c) = consistent_estimators ef in
  let var_r = Tensor.((1.0 + (s_c + c) / (1.0 - c)) * v_c) in
  let var_v = Tensor.(2.0 * (v_c ** 2.0) / (1.0 - c)) in
  let var_s = Tensor.(2.0 * (c + 2.0 * s_c) + 2.0 * ((c + s_c) ** 2.0) / (1.0 - c)) in
  (r_c, v_c, s_c, var_r, var_v, var_s)

let quadratic_loss true_value estimated_value =
  Tensor.((true_value - estimated_value) ** 2.0)

let generate_frontier ef estimator num_points =
  let (r_gmv, v_gmv, s) = Estimators.estimate ef estimator in
  let r_min = Tensor.to_float0_exn (r_gmv - Tensor.sqrt (s * v_gmv)) in
  let r_max = Tensor.to_float0_exn (r_gmv + Tensor.sqrt (s * v_gmv)) in
  let step = (r_max -. r_min) /. float_of_int (num_points - 1) in
  List.init num_points (fun i ->
    let r = r_min +. step *. float_of_int i in
    let v = Tensor.to_float0_exn (v_gmv + (Tensor.of_float1 r - r_gmv) ** 2.0 / s) in
    (r, v)
  )