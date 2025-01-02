open Torch

type t = {
  name: string;
  data: Tensor.t;
}

val create : string -> Tensor.t -> t
val mean : t -> Tensor.t
val std : t -> Tensor.t
val sharpe : t -> Tensor.t
val t_statistic : t -> Tensor.t
val p_value : t -> Tensor.t
val calculate_premium : t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t
val calculate_irr : t -> Tensor.t
val calculate_newey_west_t_stat : t -> int -> Tensor.t
val bootstrap_confidence_interval : t -> float -> int -> Tensor.t * Tensor.t
val calculate_cumulative_return : t -> Tensor.t
val calculate_drawdown : t -> Tensor.t
val calculate_sortino_ratio : t -> float -> Tensor.t
val calculate_factor_timing : t -> t -> Tensor.t * Tensor.t
val calculate_factor_timingseries : t -> t -> int -> Tensor.t * Tensor.t