open Torch

val parameter_sensitivity : 'a -> (string, float) Hashtbl.t -> ('a -> Tensor.t) -> string -> float * float -> int -> (float * Tensor.t) list
val print_sensitivity_results : string -> (float * Tensor.t) list -> unit