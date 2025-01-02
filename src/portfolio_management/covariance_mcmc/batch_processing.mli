open Torch

type batch_config = {
  min_batch_size: int;
  max_batch_size: int;
  target_memory_gb: float;
  adaptation_rate: float;
}

val optimal_batch_size : int -> batch_config -> int

val batch_mm : Tensor.t list -> int -> Tensor.t

val batch_covariance : Tensor.t -> int -> Tensor.t

val streaming_mean : (unit -> Tensor.t option) -> int -> Tensor.t

val streaming_covariance : (unit -> Tensor.t option) -> int -> Tensor.t * Tensor.t