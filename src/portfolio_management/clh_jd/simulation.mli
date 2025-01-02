open Torch
open Hedging

val simulate_hedging : 
  Hedging.hedging_strategy -> 
  int -> 
  Hedging.hedging_method ->
  Tensor.t * Tensor.t * Tensor.t

val calculate_hedging_error :
  Tensor.t -> Tensor.t -> Option.t -> float -> Tensor.t

val calculate_mean_hedging_error :
  Tensor.t -> Tensor.t -> Option.t -> Tensor.t

val calculate_var_hedging_error :
  Tensor.t -> Tensor.t -> Option.t -> float -> Tensor.t

val calculate_sharpe_ratio :
  Tensor.t -> float -> Tensor.t

val calculate_maximum_drawdown :
  Tensor.t -> Tensor.t

val print_advanced_statistics :
  Tensor.t -> Tensor.t -> Tensor.t -> Option.t -> float -> unit

val compare_hedging_methods :
  Hedging.hedging_strategy -> int -> unit