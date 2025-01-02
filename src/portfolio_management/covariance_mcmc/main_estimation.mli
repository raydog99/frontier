open Torch
open Types

type config = {
  device: Gpu_compute.device_config;
  regularization: Ill_conditioned.regularization;
  batch_config: Batch_processing.batch_config;
  n_parallel_chains: int;
}

(** Main covariance estimation function *)
val estimate_covariance :
  markov_chain ->
  Tensor.t ->
  int ->
  config ->
  distribution * float

(** Comprehensive estimation with all guarantees *)
val estimate_with_guarantees :
  markov_chain ->
  Tensor.t ->
  int ->
  config ->
  distribution * 
  Convergence_diagnostics.convergence_stats * 
  (bool * string list)