open Torch

val solve : 
  matrices:Tensor.t array -> 
  epsilon:float ->
  max_iterations:int ->
  convergence_threshold:float ->
  Types.sdp_solution

val compute_oracle : 
  psi:Tensor.t ->
  samples:Tensor.t ->
  center:Tensor.t ->
  batch_size:int ->
  values:Tensor.t * trace:float

val verify_solution :
  solution:Types.sdp_solution ->
  matrices:Tensor.t array ->
  epsilon:float ->
  bool