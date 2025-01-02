open Torch

val rect_multiply : Tensor.t -> Tensor.t -> Tensor.t
val block_multiply : Tensor.t -> Tensor.t -> int -> Tensor.t
val transpose_multiply : Tensor.t -> Tensor.t -> Tensor.t