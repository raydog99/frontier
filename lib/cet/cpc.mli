open Torch

type t

val create : int -> t
val forward : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
val infonce_loss : Tensor.t -> Tensor.t -> float -> Tensor.t
val state_dict : t -> (string * Tensor.t) list
val load_state_dict : t -> (string * Tensor.t) list -> unit