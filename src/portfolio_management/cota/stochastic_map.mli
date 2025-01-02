open Torch

val create_from_plan : Tensor.t -> (Tensor.t -> Tensor.t)
val average_maps : (Tensor.t -> Tensor.t) list -> (Tensor.t -> Tensor.t)