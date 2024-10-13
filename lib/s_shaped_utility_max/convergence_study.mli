open Torch

val convergence_study : ('a -> 'b -> int -> unit) -> ('a -> Tensor.t) -> 'a -> 'b -> int list -> (int * float * float) list
val print_convergence_results : (int * float * float) list -> unit