open Torch

type t

val predict : t -> Tensor.t -> Tensor.t
val train_linear_regression : Tensor.t -> Tensor.t -> t
val train_pls : Tensor.t -> Tensor.t -> int -> t
val train_neural_network : Tensor.t -> Tensor.t -> t