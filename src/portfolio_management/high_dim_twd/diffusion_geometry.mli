open Torch

val build_affinity_matrix : Tensor.t -> float -> Tensor.t
val normalize_affinity : Tensor.t -> Tensor.t
val build_diffusion_operator : Tensor.t -> Tensor.t
val fast_affinity_matrix : Tensor.t -> float -> Tensor.t
val stable_normalize_affinity : Tensor.t -> Tensor.t