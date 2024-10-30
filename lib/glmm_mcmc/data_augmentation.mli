open Torch
open Types

module Probit : sig
  type augmented_state = {
    params: model_params;
    v: Tensor.t;
  }
  val step : data -> augmented_state -> prior_params -> augmented_state
end

module Logistic : sig
  type augmented_state = {
    params: model_params;
    w: Tensor.t;
  }
  val step : data -> augmented_state -> prior_params -> augmented_state
end

module PX : sig
  val probit_step : data -> Probit.augmented_state -> 
                    prior_params -> Probit.augmented_state
end