open Torch
open Types

val estimate_unregularized :
  tasks:task_data list ->
  sigma_sq:float ->
  Tensor.t
(** Unregularized hyper-covariance estimation *)

val estimate_l1_regularized :
  tasks:task_data list ->
  sigma_sq:float ->
  lambda:float ->
  Tensor.t
(** L1 regularized hyper-covariance estimation *)

val estimate_correlation :
  tasks:task_data list ->
  l0:int ->
  lambda:float ->
  Tensor.t
(** Correlation-based estimation *)