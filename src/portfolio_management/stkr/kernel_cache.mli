open Torch

type t
val create : int -> t
val get_or_compute : t -> string -> (Tensor.t -> Tensor.t -> Tensor.t) -> 
    Tensor.t -> Tensor.t -> Tensor.t
val compute_key : Tensor.t -> Tensor.t -> string