open Torch

val check_bounded : Tensor.t -> bool
val check_identity_proximity : Tensor.t -> float -> bool
val compute_fourth_moment : Tensor.t -> Tensor.t