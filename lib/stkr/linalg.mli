open Torch

val eigensystem : Tensor.t -> Tensor.t * Tensor.t
val solve_conjugate_gradient : Tensor.t -> Tensor.t -> float -> int -> Tensor.t