open Torch

val estimate_bounded_covariance : 
  samples:Tensor.t ->
  epsilon:float ->
  Tensor.t

val estimate_known_covariance : 
  samples:Tensor.t ->
  epsilon:float ->
  tau:float ->
  Tensor.t

val compute_weights : 
  samples:Tensor.t ->
  center:Tensor.t ->
  epsilon:float ->
  Tensor.t