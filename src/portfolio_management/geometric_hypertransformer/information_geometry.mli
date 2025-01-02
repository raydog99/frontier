open Torch

type geometric_structure = {
  metric: Tensor.t -> Tensor.t -> Tensor.t;
  connection: Tensor.t -> Tensor.t array array;
  curvature: Tensor.t -> float;
}

val exponential_map: geometric_structure -> Tensor.t -> Tensor.t -> Tensor.t
val logarithmic_map: geometric_structure -> Tensor.t -> Tensor.t -> Tensor.t
val parallel_transport: geometric_structure -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t