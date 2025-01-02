open Factor_model
open Torch

val bootstrap_validation : Factor_model.t -> Tensor.t -> Tensor.t -> int -> float list
val k_fold_cross_validation : Factor_model.t -> Tensor.t -> Tensor.t -> int -> float list