open Torch

val refine_sqrt_epsilon : 
  current:Tensor.t ->
  samples:Tensor.t ->
  epsilon:float ->
  Tensor.t

val refine_log_epsilon : 
  current:Tensor.t ->
  samples:Tensor.t ->
  epsilon:float ->
  tau:float ->
  Tensor.t

val verify_refinement :
  estimate:Tensor.t ->
  previous:Tensor.t ->
  epsilon:float ->
  bool