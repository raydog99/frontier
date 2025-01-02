open Torch

val generate_truncated_exponential : float -> int -> float -> Tensor.t
val generate_beta : float -> float -> int -> Tensor.t
val generate_mixture_distribution : Tensor.t -> Tensor.t -> float -> int -> Tensor.t