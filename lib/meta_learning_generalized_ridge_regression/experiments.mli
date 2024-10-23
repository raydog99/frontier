open Torch
open Types

val run_unregularized_experiment :
  config:experiment_config ->
  omega:Tensor.t ->
  sigma:Tensor.t ->
  float * float * float * float * float
(** Run unregularized experiment *)

val run_l1_regularized_experiment :
  config:experiment_config ->
  omega:Tensor.t ->
  sigma:Tensor.t ->
  lambda:float ->
  float * float * float * float * float
(** Run L1 regularized experiment *)

val run_correlation_benchmark :
  config:experiment_config ->
  lambda:float ->
  float * float * float
(** Run correlation estimation benchmark *)