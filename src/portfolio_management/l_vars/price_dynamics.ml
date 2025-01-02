open Torch
open Paramter_shift

type t

val create : float -> float -> float -> ?parameter_shift:Parameter_shift.t option -> unit -> t
val simulate_price : t -> float -> float -> int -> Tensor.t