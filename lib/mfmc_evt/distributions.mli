open Torch
open Types

module type Distribution = sig
  type params
  val sample : params -> int -> (Tensor.t, mf_error) result
  val log_likelihood : params -> Tensor.t -> (Tensor.t, mf_error) result
  val fit_mle : Tensor.t -> (params, mf_error) result
  val params_to_tensor : params -> (Tensor.t, mf_error) result
  val params_from_tensor : Tensor.t -> (params, mf_error) result
end

module BivariateGaussian : Distribution with type params = distribution_params

module BivariateGumbel : Distribution with type params = distribution_params

module Bernoulli : Distribution with type params = distribution_params

module Gumbel : Distribution with type params = distribution_params