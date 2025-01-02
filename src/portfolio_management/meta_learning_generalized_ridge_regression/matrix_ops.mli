open Torch

val frobenius_norm : Tensor.t -> float
(** Compute Frobenius norm of a matrix *)

val operator_norm : Tensor.t -> float
(** Compute operator norm of a matrix *)

val cholesky_add_jitter : Tensor.t -> Tensor.t
(** Compute Cholesky decomposition with numerical stability *)