open Types
open Torch

val log_likelihood : model -> Tensor.t -> Tensor.t -> float

val optimize_mle : model -> Tensor.t -> Tensor.t -> float -> int -> model

val estimate_parameters : model -> Tensor.t -> Tensor.t -> float -> int -> model