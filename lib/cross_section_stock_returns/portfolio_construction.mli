open Torch

val hierarchical_risk_parity : Tensor.t -> float list
val robust_mean_variance_optimization : Tensor.t -> Tensor.t -> float -> float -> Tensor.t
val cdar_optimization : Tensor.t -> float -> float -> Tensor.t