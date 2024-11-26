open Torch

type market_params = {
  kappa: float;         (* κ ∈ [1/2, 1] *)
  num_stocks: int;      (* n ≥ 2 *)
  time_horizon: float;  (* T > 0 *)
}

type numerical_params = {
  time_steps: int;
  space_steps: int;
  monte_carlo_paths: int;
  error_tolerance: float;
}

type solution = {
  stocks: Tensor.t;           (* Stock price paths *)
  bessel: Tensor.t array;     (* Bessel process paths *)
  arbitrage: Tensor.t;        (* Optimal arbitrage values *)
  time_changes: float array;  (* Time changes *)
  error_metrics: error_metrics;
}

type error_metrics = {
  l2_error: float;
  max_error: float;
  relative_error: float;
  convergence_rate: float;
  confidence_intervals: (float * float) array;
}

type state = {
  stocks: Tensor.t;
  total_market: Tensor.t;
  time: float;
}

type market_params = {
  kappa: float;         (* κ ∈ [1/2, 1] *)
  num_stocks: int;      (* n ≥ 2 *)
  time_horizon: float;  (* T > 0 *)
}

type numerical_params = {
  time_steps: int;
  space_steps: int;
  monte_carlo_paths: int;
  error_tolerance: float;
}

type state = {
  stocks: Tensor.t;
  total_market: Tensor.t;
  time: float;
}

type error_metrics = {
  l2_error: float;
  max_error: float;
  relative_error: float;
  convergence_rate: float;
  confidence_intervals: (float * float) array;
}

type solution = {
  stocks: Tensor.t;           (* Stock price paths *)
  bessel: Tensor.t array;     (* Bessel process paths *)
  arbitrage: Tensor.t;        (* Optimal arbitrage values *)
  time_changes: float array;  (* Time changes *)
  error_metrics: error_metrics;
}

val generate_brownian_increments : 
  num_steps:int -> 
  num_stocks:int -> 
  device:Device.t -> 
  float -> 
  Tensor.t

val simulate_bessel_process : 
  dimension:int -> 
  initial_value:float -> 
  num_steps:int -> 
  float -> 
  Tensor.t

val solve : market_params -> numerical_params -> state -> solution

val validate_solution : solution -> bool

val simulate_market : market_params -> numerical_params -> state -> state

val compute_optimal_arbitrage : market_params -> numerical_params -> state -> Tensor.t

val compute_error_metrics : 
  numerical_sol:Tensor.t -> 
  bessel_paths:Tensor.t array -> 
  arbitrage:Tensor.t -> 
  error_metrics

val create_initial_state : float array -> state
val time_change : state -> float -> float
val interpolate : state -> state -> float -> state