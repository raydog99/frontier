open Torch

type t

val create : int -> int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val parameters : t -> Tensor.t list