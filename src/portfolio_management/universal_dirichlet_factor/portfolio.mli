open Torch

val generate_factor_market : int -> int -> Tensor.t
val generate_multi_factor_market : int -> int -> int -> Tensor.t
val create_crp : int -> Tensor.t
val sample_dirichlet : Tensor.t -> Tensor.t
val create_dfp : int -> float -> Tensor.t
val portfolio_return : Tensor.t -> Tensor.t -> Tensor.t
val growth_rate : Tensor.t -> Tensor.t -> Tensor.t
val simulate_dfp : int -> int -> int -> float -> Tensor.t
val monte_carlo_dfp : int -> int -> int -> float -> int -> Tensor.t * Tensor.t
val approximate_target_portfolio : Tensor.t -> int -> float -> int -> Tensor.t
val tracking_error : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val sharpe_ratio : Tensor.t -> Tensor.t -> float -> Tensor.t
val maximum_drawdown : Tensor.t -> Tensor.t -> Tensor.t
val moving_average : Tensor.t -> int -> Tensor.t
val compare_dfp_to_equal_weight : int -> int -> int -> float -> int -> float * float * float * float
val confidence_interval : Tensor.t -> Tensor.t -> float -> int -> float * float
val rebalance_portfolio : Tensor.t -> Tensor.t -> float -> Tensor.t * Tensor.t
val backtest : (int -> Tensor.t) -> Tensor.t -> Tensor.t -> int -> float -> (float * Tensor.t) list
val annualized_return : float -> float -> float -> float
val annualized_volatility : Tensor.t -> int -> float
val main : unit -> unit