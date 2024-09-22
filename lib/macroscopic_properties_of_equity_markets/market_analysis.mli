open Torch

val calculate_market_weights : float array -> float array
val calculate_capital_distribution : float array -> float array
val calculate_shannon_entropy : float array -> float
val calculate_excess_growth_rate : float array -> float array -> float
val calculate_rank_volatility : float array array -> float array
val calculate_rank_transition_probabilities : float array array -> float array array
val calculate_rank_switching_intensity : float array array -> float array
val calculate_market_weights_tensor : Tensor.t -> Tensor.t
val calculate_capital_distribution_tensor : Tensor.t -> Tensor.t
val calculate_rank_volatility_tensor : Tensor.t -> Tensor.t
val calculate_rank_transition_probabilities_tensor : Tensor.t -> Tensor.t
val get_rank : float array -> int -> int
val get_top_k_stocks : float array -> int -> int list