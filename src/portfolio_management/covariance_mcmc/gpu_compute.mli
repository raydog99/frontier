open Torch

type device_config = {
  use_gpu: bool;
  device_id: int;
  precision: [`Float | `Double];
}

val get_device : device_config -> Device.t

val to_device : Tensor.t -> device_config -> Tensor.t

(** GPU-accelerated batch matrix multiplication *)
val gpu_batch_mm : Tensor.t list -> device_config -> Tensor.t

(** GPU-accelerated covariance estimation *)
val gpu_covariance : Tensor.t -> device_config -> Tensor.t

(** GPU-accelerated SVD computation *)
val gpu_svd : Tensor.t -> device_config -> Tensor.t * Tensor.t * Tensor.t