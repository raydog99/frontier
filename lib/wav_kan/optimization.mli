open Torch

type optimizer_config = {
  learning_rate: float;
  momentum: float option;
  beta1: float option;
  beta2: float option;
  weight_decay: float;
  grad_clip: float option;
}

type t

val create : optimizer_config -> WavKANNetwork.t -> t
val step : t -> unit
val zero_grad : t -> unit

module Regularization : sig
  type regularizer_type =
    | Smoothness of float
    | Sparsity of float
    | Energy of float
    | Orthogonality of float

  val compute_penalty : Tensor.t list -> regularizer_type list -> Tensor.t