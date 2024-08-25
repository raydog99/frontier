open Torch

type t

val create : int -> int -> int -> float -> t
val train : t -> Tensor.t -> Tensor.t -> int -> unit
val predict : t -> Tensor.t -> Tensor.t
val save : t -> string -> unit
val load : t -> string -> unit
