open Torch
open Ctmc

val delta : t -> float -> float -> float -> int -> int -> Tensor.t
val replicate_arrow_debreu : t -> float -> float -> float -> int -> int -> float