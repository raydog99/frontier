open Torch
open Types

val mean : Tensor.t -> Tensor.t

val covariance : Tensor.t -> Tensor.t -> Tensor.t

val spectral_gap : Tensor.t -> float

(** Estimate trace using Hutchinson's estimator *)
val estimate_trace : Tensor.t -> int -> float