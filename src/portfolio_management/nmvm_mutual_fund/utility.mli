open Torch

type t = Tensor.t -> Tensor.t

val exponential : float -> t
val power : float -> t
val log : t
val quadratic : float -> float -> t
val sahara : float -> float -> float -> t
val crra : float -> t
val cara : float -> t
val expected_utility : t -> Tensor.t -> Tensor.t