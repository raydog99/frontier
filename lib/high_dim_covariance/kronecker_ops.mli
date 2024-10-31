open Torch

val matrix_vector_product : Tensor.t -> Tensor.t -> int -> Tensor.t
val transpose_product : Tensor.t -> Tensor.t -> int -> Tensor.t
val efficient_outer_product : Tensor.t -> Tensor.t -> Tensor.t