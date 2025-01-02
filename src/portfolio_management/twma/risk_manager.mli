type t

val create : ?max_drawdown:float -> ?var_threshold:float -> ?es_threshold:float ->
             ?volatility_target:float -> ?max_leverage:float -> ?correlation_threshold:float -> unit -> t
val update_historical_returns : t -> float array -> unit
val apply_risk_management : t -> float array -> float array array -> float array
