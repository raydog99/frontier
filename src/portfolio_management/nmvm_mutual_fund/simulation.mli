type result = {
  portfolio: Portfolio.t;
  returns: float array;
  wealth: float array;
  performance: Performance.t;
}

type rebalancing_strategy =
  | NoRebalancing
  | PeriodicRebalancing of int
  | ThresholdRebalancing of float

val run : 
  Nmvm.t -> 
  Portfolio.t -> 
  float ->  (* initial wealth *)
  float ->  (* risk-free rate *)
  int ->    (* number of periods *)
  rebalancing_strategy ->
  Optimizer.strategy ->
  result

val compare_strategies : 
  Nmvm.t -> 
  (string * Optimizer.strategy) list -> 
  float ->  (* initial wealth *)
  float ->  (* risk-free rate *)
  int ->    (* number of periods *)
  int ->    (* number of simulations *)
  rebalancing_strategy ->
  (string * result) list