open Torch

val poincare_distance : Tensor.t -> Tensor.t -> float
val frechet_mean : Tensor.t -> Tensor.t
val stable_poincare_distance : Tensor.t -> Tensor.t -> float