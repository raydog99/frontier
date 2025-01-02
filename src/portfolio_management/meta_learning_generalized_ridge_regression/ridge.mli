open Torch
open Types

val estimate : 
  x:Tensor.t -> 
  y:Tensor.t -> 
  a:Tensor.t -> 
  n_l:int -> 
  Tensor.t
(** Compute ridge regression estimate *)

val oracle_risk :
  x_new:Tensor.t ->
  y_new:Tensor.t ->
  beta:Tensor.t ->
  omega:Tensor.t ->
  float
(** Compute oracle predictive risk *)

val compute_asymptotic_risk :
  x:Tensor.t ->
  y:Tensor.t ->
  omega:Tensor.t ->
  sigma_sq:float ->
  gamma:float ->
  float
(** Compute asymptotic risk *)