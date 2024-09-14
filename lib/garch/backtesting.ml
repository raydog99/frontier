open Torch
open Types
open Garch_core

type backtest_result = {
  model: garch_model;
  mse: float;
  mae: float;
  var_violations: float;
  es_violations: float;
  likelihood_ratio: float;
}

let kupiec_test violations total confidence_level =
  let expected_violations = float_of_int total *. (1.0 -. confidence_level) in
  let observed_violations = float_of_int violations in
  let likelihood_ratio = 2.0 *. (observed_violations *. log (observed_violations /. expected_violations) +.
                                 (float_of_int total -. observed_violations) *. log ((float_of_int total -. observed_violations) /. (float_of_int total -. expected_violations))) in
  let p_value = 1.0 -. Statistics.chi_square_cdf likelihood_ratio 1 in
  (likelihood_ratio, p_value)

let backtest_model model params returns confidence_level window_size =
  let n = Tensor.shape returns |> List.hd in
  let var_violations = ref 0 in
  let es_violations = ref 0 in
  let squared_errors = ref [] in
  let abs_errors = ref [] in

  for t = window_size to n - 1 do
    let train_returns = Tensor.slice returns ~dim:0 ~start:(t - window_size) ~end:(Some t) in
    let forecasted_vol = forecast_volatility model params train_returns 1 |> Tensor.get_float1 0 in
    let realized_vol = Tensor.get_float1 returns t ** 2.0 in
    squared_errors := (forecasted_vol -. realized_vol) ** 2.0 :: !squared_errors;
    abs_errors := abs_float (forecasted_vol -. realized_vol) :: !abs_errors;

    let var = Statistics.normal_ppf confidence_level *. sqrt forecasted_vol in
    let es = Statistics.normal_pdf (Statistics.normal_ppf confidence_level) /. (1.0 -. confidence_level) *. sqrt forecasted_vol in

    if Tensor.get_float1 returns t < -.var then incr var_violations;
    if Tensor.get_float1 returns t < -.es then incr es_violations;
  done;

  let mse = List.fold_left (+.) 0.0 !squared_errors /. float_of_int (List.length !squared_errors) in
  let mae = List.fold_left (+.) 0.0 !abs_errors /. float_of_int (List.length !abs_errors) in
  let var_violation_rate = float_of_int !var_violations /. float_of_int (n - window_size) in
  let es_violation_rate = float_of_int !es_violations /. float_of_int (n - window_size) in
  let likelihood_ratio, _ = kupiec_test !var_violations (n - window_size) confidence_level in

  { model; mse; mae; var_violations = var_violation_rate; es_violations = es_violation_rate; likelihood_ratio }

let compare_models data confidence_level window_size =
  let returns = Tensor.(log (slice data.closes ~dim:0 ~start:1 ~end:None) - log (slice data.closes ~dim:0 ~start:0 ~end:(Some (-1)))) in
  let models = [GARCH; EGARCH; GJR_GARCH] in
  List.map (fun model ->
    let params = estimate_garch_parameters_generic model returns ~max_iter:1000 ~learning_rate:0.01 in
    backtest_model model params returns confidence_level window_size
  ) models