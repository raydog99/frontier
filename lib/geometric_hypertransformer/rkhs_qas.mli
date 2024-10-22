open Torch

type kernel = {
  k: Tensor.t -> Tensor.t -> float;
  grad_k: Tensor.t -> Tensor.t -> Tensor.t;
  feature_map: Tensor.t -> Tensor.t option;
}

type point = {
  value: Tensor.t;
  kernel: kernel;
  norm: float;
}

include QAS_SPACE with type t = point

val gaussian_kernel: float -> kernel
val create_mixing: float -> int -> mixing_function