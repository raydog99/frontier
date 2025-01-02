open Torch
open Types

type regularization = 
  | Tikhonov of float
  | Truncated of float
  | Adaptive of float

val estimate_condition_number : Tensor.t -> float

val regularize : Tensor.t -> regularization -> Gpu_compute.device_config -> Tensor.t

val robust_covariance_estimation :
  Tensor.t ->
  regularization ->
  Gpu_compute.device_config ->
  distribution