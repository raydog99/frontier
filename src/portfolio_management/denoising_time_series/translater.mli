open Torch

type t

val create : int -> int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val train : t -> Tensor.t -> Tensor.t -> float -> Tensor.t