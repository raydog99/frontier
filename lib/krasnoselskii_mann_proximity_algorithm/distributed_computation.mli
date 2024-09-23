open Torch

val split_workload : int list -> Tensor.t -> (int * Tensor.t) list
val gather_results : Tensor.t list -> Tensor.t