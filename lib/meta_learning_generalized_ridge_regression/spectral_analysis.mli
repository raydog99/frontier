open Torch
open Types

val compute_esd : Tensor.t -> spectral_distribution
(** Compute empirical spectral distribution *)

val compute_lsd : 
  matrix:Tensor.t -> 
  gamma:float -> 
  spectral_distribution
(** Compute limiting spectral distribution *)

val verify_convergence :
  matrix:Tensor.t ->
  gamma:float ->
  num_points:int ->
  float list
(** Verify spectral convergence *)