open Torch

type order = Zero | First | Second

val kernel_zero : Tensor.t -> Tensor.t -> Tensor.t
val kernel_first : Tensor.t -> Tensor.t -> Tensor.t
val kernel_second : Tensor.t -> Tensor.t -> Tensor.t
val kernel : order -> Tensor.t -> Tensor.t -> Tensor.t
val kernel_function : order -> Tensor.t -> (int -> int -> float)