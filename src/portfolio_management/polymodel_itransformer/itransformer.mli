open Torch

type t

val create : int -> int -> int -> int -> t
val forward : t -> Tensor.t -> Tensor.t
val loss_fn : Tensor.t -> Tensor.t -> Tensor.t
val train : t -> Dataset.t -> float -> int -> unit
val predict : t -> Tensor.t -> Tensor.t
val save_model : t -> string -> unit
val load_model : string -> int -> int -> int -> int -> t