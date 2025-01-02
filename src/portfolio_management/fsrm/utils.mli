open Torch

val arctan_transform : Tensor.t -> Tensor.t
val inverse_arctan_transform : Tensor.t -> Tensor.t
val integrate : (float -> float) -> float -> float -> int -> float
val linspace : float -> float -> int -> Tensor.t
val find_root : (float -> float) -> float -> float -> float -> float
val confidence_interval : Tensor.t -> float -> float * float