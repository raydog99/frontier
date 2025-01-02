open Torch

exception Invalid_input_dimension of string

type t

val create : int -> int -> float -> t
val select_action : t -> Tensor.t -> Tensor.t
val update : t -> Tensor.t * Tensor.t * float * Tensor.t * bool -> unit
val get_alpha : t -> float
val save : t -> string -> unit
val load : t -> string -> unit
