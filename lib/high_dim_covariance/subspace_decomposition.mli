open Torch

type partition = {
  high_eigenspace: Tensor.t;
  medium_eigenspace: Tensor.t;
  low_eigenspace: Tensor.t;
}

val decompose : 
  matrix:Tensor.t -> 
  epsilon:float -> 
  partition

val project_samples : 
  samples:Tensor.t -> 
  subspace:Tensor.t -> 
  Tensor.t

val combine_estimates : 
  high:Tensor.t ->
  medium:Tensor.t ->
  cross_terms:Tensor.t ->
  partition:partition ->
  epsilon:float ->
  Tensor.t
