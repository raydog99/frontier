open Torch

(* Normal space density rescaling *)
val rescale_transition : float -> Tensor.t -> Tensor.t
val rescale_measurement : float -> Tensor.t -> Tensor.t

(* Log space density rescaling *)
val rescale_transition_log : float -> Tensor.t -> Tensor.t
val rescale_measurement_log : float -> Tensor.t -> Tensor.t

(* Information bottleneck optimization *)
val optimize_bottleneck : 
  prior:Tensor.t ->
  transition_prob:Tensor.t ->
  measurement_prob:Tensor.t ->
  lambda:float ->
  Tensor.t