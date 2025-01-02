open Torch
open Sde

type t

val create : int -> int -> t
val forward : t -> Tensor.t -> Tensor.t -> Tensor.t
val loss : t -> SDE -> Tensor.t -> float -> Tensor.t
val to_serialize : t -> Serialize.t
val of_serialize : Serialize.t -> t