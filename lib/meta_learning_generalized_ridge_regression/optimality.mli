open Torch
open Types

val compute_optimality_gradient :
  omega:Tensor.t ->
  sigma:Tensor.t ->
  gamma:float ->
  sigma_sq:float ->
  Tensor.t
(** Compute Riemannian gradient for optimality *)

val verify_pd_hessian :
  point:Tensor.t ->
  gamma:float ->
  sigma_sq:float ->
  bool
(** Verify positive definiteness of Hessian *)

val compute_optimal_risk :
  omega:Tensor.t ->
  sigma:Tensor.t ->
  gamma:float ->
  sigma_sq:float ->
  float
(** Compute optimal risk value *)