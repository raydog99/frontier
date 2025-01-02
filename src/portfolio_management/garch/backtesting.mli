open Torch
open Types

type backtest_result = {
  model: garch_model;
  mse: float;
  mae: float;
  var_violations: float;
  es_violations: float;
  likelihood_ratio: float;
}

val kupiec_test : int -> int -> float -> float * float
val backtest_model : garch_model -> float * float * float * float -> Tensor.t -> float -> int -> backtest_result
val compare_models : historical_data -> float -> int -> backtest_result list