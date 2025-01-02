open Torch

module type Model = sig
  type real_state
  type virtual_state
  type real_measurement 
  type virtual_measurement
  type params

  val virtual_transition : real_state -> virtual_state Tensor.t -> params -> virtual_state Tensor.t
  val virtual_measurement : real_state -> virtual_measurement Tensor.t -> params -> virtual_measurement Tensor.t
  val real_to_virtual_state : real_state -> virtual_state
  val real_to_virtual_measurement : real_measurement -> virtual_measurement
end

module type StateEstimator = sig
  type state
  val estimate_state : state -> Tensor.t
  val estimate_uncertainty : state -> Tensor.t option
end

type distance_metric = 
  | Quadratic
  | KLDivergence

type threshold_dist = 
  | Exponential of float