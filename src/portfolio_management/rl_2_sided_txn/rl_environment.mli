open Torch

exception Invalid_action of string
exception Environment_error of string

type t

val create : Tensor.t -> float -> float -> t
val reset : t -> unit
val step : t -> Tensor.t -> float * float * bool
val get_state : t -> Tensor.t
