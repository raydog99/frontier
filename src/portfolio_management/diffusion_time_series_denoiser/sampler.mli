open Torch
open Score_network
open Sde

val predictor_step : SDE -> Score_network.t -> Tensor.t -> float -> float -> Tensor.t
val corrector_step : SDE -> Score_network.t -> Tensor.t -> float -> Tensor.t
val sample : SDE -> Score_network.t -> Tensor.t -> int -> Tensor.t