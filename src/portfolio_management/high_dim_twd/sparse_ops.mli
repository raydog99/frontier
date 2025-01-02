open Torch

type sparse_tensor = {
  indices: Tensor.t;
  values: Tensor.t;
  size: int array;
}

val create_sparse : Tensor.t -> Tensor.t -> int array -> sparse_tensor
val to_dense : sparse_tensor -> Tensor.t
val from_dense : Tensor.t -> float -> sparse_tensor
val sparse_mm : sparse_tensor -> sparse_tensor -> sparse_tensor