open Torch
open Types

type spatial_model = {
  data: data;
  spatial_params: Spatial.spatial_params;
  model_params: model_params;
}

val create : data -> Spatial.spatial_params -> spatial_model

module MCMC : sig
  val mala_step : spatial_model -> mcmc_state -> float -> mcmc_state
  val hmc_step : spatial_model -> HMC.hmc_state -> float -> int -> model_params
end

module Correlation : sig
  val exponential : Tensor.t -> float -> Tensor.t
  val gaussian : Tensor.t -> float -> Tensor.t
  val matern : Tensor.t -> float -> float -> Tensor.t
end