open Torch

val compute_bounds : 
  estimate:Tensor.t -> 
  true_cov:Tensor.t -> 
  Types.error_bound

val track_convergence : 
  current:Tensor.t ->
  previous:Tensor.t ->
  epsilon:float ->
  bool * float

val verify_error_rate : 
  history:Tensor.t list ->
  epsilon:float ->
  bool