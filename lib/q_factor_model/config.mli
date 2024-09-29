type t = {
  start_year: int;
  end_year: int;
  confidence_level: float;
  n_bootstrap: int;
  newey_west_lags: int;
  train_ratio: float;
  rolling_window_size: int;
  rolling_window_step: int;
  risk_free_rate: float;
}

val default : t

val create : 
  ?start_year:int ->
  ?end_year:int ->
  ?confidence_level:float ->
  ?n_bootstrap:int ->
  ?newey_west_lags:int ->
  ?train_ratio:float ->
  ?rolling_window_size:int ->
  ?rolling_window_step:int ->
  ?risk_free_rate:float ->
  unit -> t