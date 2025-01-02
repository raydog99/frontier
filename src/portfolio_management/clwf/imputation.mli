open Torch

module PotentialFunction : sig
  type t
  val create : vae:VAE.t -> sigma_p:float -> t
  val compute_gradient : t -> Tensor.t -> Tensor.t
  val compute_energy : t -> Tensor.t -> Tensor.t
end

module RaoBlackwellizedSampler : sig
  type t
  val create : flow_model:FlowNetwork.t -> vae:VAE.t -> config:model_config -> t
  val step : t -> state:Tensor.t -> conditional:Tensor.t -> time:float -> Tensor.t
  val generate_trajectories : t -> conditional:Tensor.t -> n_trajectories:int -> Tensor.t list
end

module Imputer : sig
  type t
  val create : flow:FlowNetwork.t -> vae:VAE.t option -> config:model_config -> t
  val impute : t -> time_series:TimeSeries.t -> n_trajectories:int -> Tensor.t
end