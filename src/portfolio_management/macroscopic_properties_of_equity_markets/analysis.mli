open Torch

val calculate_diversity_measure : float -> float array -> float
val calculate_excess_growth_rate_matrix : float array array -> float array array
val estimate_rank_based_drift : float array array -> float array
val estimate_rank_based_volatility : float array array -> float array
val calculate_leakage : float array array -> int -> float
val calculate_intrinsic_volatility : float array array -> float
val calculate_cross_sectional_momentum : Tensor.t -> int -> Tensor.t
val calculate_factor_exposures : Tensor.t -> Tensor.t -> Tensor.t
val estimate_optimal_holding_period : Tensor.t -> int -> float