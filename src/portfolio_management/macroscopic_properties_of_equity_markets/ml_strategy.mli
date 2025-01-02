open Torch

val train_model : Tensor.t -> Tensor.t -> unit
val predict : Tensor.t -> Tensor.t
val calculate_ml_portfolio : Tensor.t -> Tensor.t -> Tensor.t