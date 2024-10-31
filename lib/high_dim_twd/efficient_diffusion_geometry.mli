open Torch

val build_affinity_matrix_streaming : 
  Tensor.t -> float -> Sparse_ops.sparse_tensor array
val normalize_affinity_streaming : 
  Sparse_ops.sparse_tensor array -> Sparse_ops.sparse_tensor list