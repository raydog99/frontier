type error_metrics = {
  rmse: float;
  mae: float;
  mape: float;
}

val calculate_error_metrics : (float * float) list -> (float * float) list -> error_metrics