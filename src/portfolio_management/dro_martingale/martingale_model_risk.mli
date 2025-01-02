open Torch

type model_type = [`BlackScholes | `Bachelier]

type analysis_result = {
  model : model_type;
  sigma : float;
  t : float;
  sensitivity : Tensor.t;
  adapted_sensitivity : Tensor.t;
  forward_start_sensitivity : Tensor.t;
  forward_start_adapted_sensitivity : Tensor.t;
  martingale_sensitivity : Tensor.t;
  adapted_martingale_sensitivity : Tensor.t;
}

type hedging_strategy = {
  dynamic_hedge : Tensor.t;
  semi_static_hedge : Tensor.t;
}

type optimal_stopping_result = {
  value : Tensor.t;
  stopping_time : Tensor.t;
  sensitivity : Tensor.t;
  adapted_sensitivity : Tensor.t;
  formula_sensitivity : Tensor.t;
}

val analyze_model : model_type -> Tensor.t -> Tensor.t -> analysis_result

val compare_models : float list -> float list -> analysis_result list list list

val optimal_hedging_strategy : 
  (Tensor.t -> Tensor.t -> Tensor.t) -> 
  Tensor.t -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> 
  hedging_strategy

val analyze_optimal_stopping : 
  (Tensor.t -> Tensor.t) -> 
  (Tensor.t -> Tensor.t) -> 
  Tensor.t -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> 
  optimal_stopping_result

val run_error_analysis : model_type -> float -> float -> int -> unit

val print_analysis : analysis_result -> unit