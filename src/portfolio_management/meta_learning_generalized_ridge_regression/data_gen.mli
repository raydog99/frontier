open Torch
open Types

val generate_toeplitz :
  dim:int ->
  a:float ->
  b:float ->
  Tensor.t
(** Generate Toeplitz matrix *)

val generate_task_data :
  dim:int ->
  num_samples:int ->
  omega:Tensor.t ->
  sigma:Tensor.t ->
  sigma_sq:float ->
  task_data
(** Generate synthetic task data *)

val generate_tasks :
  config:experiment_config ->
  omega:Tensor.t ->
  sigma:Tensor.t ->
  task_data list
(** Generate multiple tasks *)