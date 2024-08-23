open Torch

type loss_type =
  | Linear
  | Quadratic
  | Exponential
  | Custom of (Tensor.t -> Tensor.t -> Tensor.t)

val create_loss_function : loss_type -> Types.loss_function