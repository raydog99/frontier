open Torch

type loss_type =
  | Linear
  | Quadratic
  | Exponential
  | Custom of (Tensor.t -> Tensor.t -> Tensor.t)

let create_loss_function = function
  | Linear -> fun y z -> Tensor.(y * z)
  | Quadratic -> fun y z -> Tensor.(pow (y * z) (scalar 2.))
  | Exponential -> fun y z -> Tensor.(exp (y * z))
  | Custom f -> f