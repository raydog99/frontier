open Torch

type t

val create : int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val state_dict : t -> (string * (string * Tensor.t) list) list
val load_state_dict : t -> (string * (string * Tensor.t) list) list -> unit