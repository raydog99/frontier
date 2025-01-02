open Torch

val calculate_diversity_weighted_portfolio : float -> float array -> float array
val calculate_portfolio_return : float array -> float array -> float array -> float
val calculate_equal_weight_portfolio : float array -> float array
val calculate_market_cap_weighted_portfolio : float array -> float array
val calculate_momentum_portfolio : float array array -> int -> float array -> float array
val calculate_sharpe_ratio : float array -> float -> float
val calculate_maximum_drawdown : float array -> float
val calculate_diversity_trend_portfolio : float array array -> float array -> float array
val calculate_volatility_responsive_portfolio : float array array -> float array -> float array
val calculate_rank_momentum_portfolio : float array array -> int -> float array -> float array
val calculate_cross_sectional_momentum_portfolio : Tensor.t -> int -> Tensor.t -> Tensor.t
val calculate_factor_tilted_portfolio : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val calculate_optimal_holding_portfolio : Tensor.t -> Tensor.t -> Tensor.t
val calculate_ml_portfolio : Tensor.t -> Tensor.t -> Tensor.t