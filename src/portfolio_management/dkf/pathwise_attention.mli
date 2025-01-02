open Torch

type t

val create : int -> int -> int -> int -> int -> device:Device.t -> t

val forward : t -> Tensor.t -> Tensor.t

val parameters : t -> Tensor.t list