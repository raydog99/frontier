open Torch

type t

val create : int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val state_dict : t -> (string * Tensor.t) list
val load_state_dict : t -> (string * Tensor.t) list -> unit