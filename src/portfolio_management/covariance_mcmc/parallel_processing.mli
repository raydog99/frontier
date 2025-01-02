open Torch
open Types

type task =
  | SampleGeneration of (unit -> Tensor.t)
  | CovarianceComputation of (Tensor.t -> Tensor.t)
  | ChainUpdate of (Tensor.t -> Tensor.t)

type worker_pool = {
  tasks: task Queue.t;
  results: Tensor.t Queue.t;
  n_workers: int;
}

val create_worker_pool : int -> worker_pool

val parallel_chain_execution :
  markov_chain ->
  Tensor.t ->
  int ->
  int ->
  Gpu_compute.device_config ->
  Tensor.t list

val parallel_covariance_estimation :
  markov_chain ->
  Tensor.t ->
  int ->
  int ->
  float ->
  float ->
  distribution