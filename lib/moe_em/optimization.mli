open Torch
open Types

module EM : sig
  val e_step : model -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
  val m_step : model -> Tensor.t -> Tensor.t -> (Tensor.t * Tensor.t) -> model
  val train : ?config:optimization_config -> model -> Tensor.t -> Tensor.t -> model * optimization_state
end

module MirrorDescent : sig
  val compute_kl_divergence : model -> model -> Tensor.t -> float
  val compute_gradients : model -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
  val compute_regularized_loss : model -> Tensor.t -> Tensor.t -> float
  val train : ?config:optimization_config -> model -> Tensor.t -> Tensor.t -> model * optimization_state
end