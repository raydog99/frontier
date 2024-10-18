open Torch
open Types

val generate_random_assets : int -> int -> asset list
val tensor_to_list : Tensor.t -> float list
val print_tensor_statistics : string -> Tensor.t -> unit