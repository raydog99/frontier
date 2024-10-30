open Torch

type spatial_params = {
  coords: Tensor.t;
  rho: Tensor.t;
  sigma2: Tensor.t;
  nu: float option;
}

val matern_correlation : Tensor.t -> float -> Tensor.t -> Tensor.t
val distance_matrix : Tensor.t -> Tensor.t
val sample_effects : spatial_params -> Tensor.t