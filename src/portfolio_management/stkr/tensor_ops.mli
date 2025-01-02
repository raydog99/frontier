open Torch

val chunked_matmul : Tensor.t -> Tensor.t -> int -> Tensor.t
val efficient_eigensystem : Tensor.t -> int -> Tensor.t * Tensor.t