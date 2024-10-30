open Torch
open Types

module Probit : sig
  val link : Tensor.t -> Tensor.t
  val inverse_link : Tensor.t -> Tensor.t
  val log_likelihood : data -> model_params -> float
end

module Logistic : sig
  val link : Tensor.t -> Tensor.t
  val inverse_link : Tensor.t -> Tensor.t
  val log_likelihood : data -> model_params -> float
end

module Poisson : sig
  val link : Tensor.t -> Tensor.t
  val inverse_link : Tensor.t -> Tensor.t
  val log_likelihood : data -> model_params -> float
end