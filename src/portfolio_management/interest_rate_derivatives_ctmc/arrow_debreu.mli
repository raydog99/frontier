open Torch
open Ctmc

val price : t -> float -> float -> Tensor.t
val price_at_state : t -> float -> float -> int -> Tensor.t