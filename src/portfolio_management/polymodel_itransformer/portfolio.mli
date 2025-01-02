open Torch

val construct_portfolio : Tensor.t -> Tensor.t -> float -> Tensor.t
val rebalance_portfolio : Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t * Tensor.t
val calculate_portfolio_return : Tensor.t -> Tensor.t -> Tensor.t
val calculate_portfolio_risk : Tensor.t -> Tensor.t -> Tensor.t
val backtest_portfolio : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float -> float
val calculate_portfolio_metrics : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t