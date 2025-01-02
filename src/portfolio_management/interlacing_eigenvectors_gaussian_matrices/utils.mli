open Torch

val assert_positive_float : string -> float -> unit
val assert_in_range : string -> float -> float -> float -> unit
val integrate : (float -> float) -> float -> float -> float -> float
val find_root : (float -> float) -> float -> float -> float -> float
val parallel_map : ('a -> 'b) -> 'a list -> 'b list Lwt.t
val safe_div : Tensor.t -> Tensor.t -> Tensor.t