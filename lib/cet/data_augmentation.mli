open Torch

val add_gaussian_noise : Tensor.t -> float -> Tensor.t
val random_scale : Tensor.t -> float * float -> Tensor.t
val time_warp : Tensor.t -> int -> Tensor.t
val augment_tensor : Tensor.t -> Tensor.t