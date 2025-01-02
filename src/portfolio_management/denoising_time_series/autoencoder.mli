open Torch

type t

val create : int -> t
val forward : t -> Tensor.t -> Tensor.t * Tensor.t
val train : t -> Tensor.t -> float -> Tensor.t * Tensor.t