open Torch

val mean_variance : Tensor.t -> Tensor.t -> float array
val minimum_variance : Tensor.t -> float array
val maximum_sharpe_ratio : Tensor.t -> Tensor.t -> float -> float array
val risk_parity : Tensor.t -> float array
val constrained_optimization : (float array -> float) -> (float array -> bool) list -> float array -> float array