open Torch
open Types
open Garch_core

type diagnostic_result = {
  lb_stat: float;
  lb_p_value: float;
  jb_stat: float;
  jb_p_value: float;
  adf_stat: float;
  adf_p_value: float;
}

let ljung_box_test residuals lags =
  let n = Tensor.shape residuals |> List.hd in
  let acf = Tensor.zeros [lags] in
  for k = 1 to lags do
    let r_k = Tensor.(mean (slice residuals ~dim:0 ~start:k ~end:n * slice residuals ~dim:0 ~start:0 ~end:(n-k))) in
    let r_0 = Tensor.var residuals in
    Tensor.set acf [k-1] Tensor.(r_k / r_0);
  done;
  let q = Tensor.(sum ((Scalar.f (float_of_int n * (n + 2))) * 
                       (acf ** (Scalar.f 2.0)) / 
                       (Scalar.f (float_of_int n - Tensor.arange 1 (lags+1) ~start:1 ~dtype:Tensor.Float)))) in
  let p_value = 1.0 -. Statistics.chi_square_cdf (Tensor.to_float0_exn q) lags in
  (Tensor.to_float0_exn q, p_value)

let jarque_bera_test residuals =
  let n = Tensor.shape residuals |> List.hd in
  let mean = Tensor.mean residuals in
  let std = Tensor.std residuals ~unbiased:true in
  let skewness = Tensor.(mean ((residuals - mean) ** (Scalar.f 3.0)) / (std ** (Scalar.f 3.0))) in
  let kurtosis = Tensor.(mean ((residuals - mean) ** (Scalar.f 4.0)) / (std ** (Scalar.f 4.0))) in
  let jb = Tensor.(Scalar.f (float_of_int n / 6.0) * (skewness ** (Scalar.f 2.0) + Scalar.f 0.25 * ((kurtosis - Scalar.f 3.0) ** (Scalar.f 2.0)))) in
  let p_value = 1.0 -. Statistics.chi_square_cdf (Tensor.to_float0_exn jb) 2 in
  (Tensor.to_float0_exn jb, p_value)

let adf_test series =
  let n = Tensor.shape series |> List.hd in
  let y = Tensor.(slice series ~dim:0 ~start:1 ~end:None) in
  let x = Tensor.(slice series ~dim:0 ~start:0 ~end:(Some (n-1))) in
  let dy = Tensor.(y - x) in
  let x_with_const = Tensor.cat [Tensor.ones [n-1; 1]; Tensor.reshape x [n-1; 1]] ~dim:1 in
  let beta = Tensor.(matmul (pinverse x_with_const) (reshape dy [n-1; 1])) in
  let residuals = Tensor.(dy - matmul x_with_const beta) in
  let se = Tensor.(sqrt (sum (residuals ** (Scalar.f 2.0)) / Scalar.f (float_of_int (n - 3)))) in
  let t_stat = Tensor.((beta.$(1) - Scalar.f 1.0) / (se * sqrt (Scalar.f (float_of_int (n - 1))))) in
  let p_value = 1.0 -. Statistics.student_t_cdf (Tensor.to_float0_exn t_stat) (n - 3) in
  (Tensor.to_float0_exn t_stat, p_value)

let run_diagnostics model params returns =
  let residuals = Tensor.(returns / sqrt (forecast_volatility model params returns (Tensor.shape returns |> List.hd))) in
  let lb_stat, lb_p_value = ljung_box_test residuals 10 in
  let jb_stat, jb_p_value = jarque_bera_test residuals in
  let adf_stat, adf_p_value = adf_test returns in
  { lb_stat; lb_p_value; jb_stat; jb_p_value; adf_stat; adf_p_value }