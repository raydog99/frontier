open Torch

val batch_pairwise_distances : 
  Tensor.t -> Tensor.t -> Config.t -> Tensor.t
val parallel_sparse_mm : 
  Sparse_ops.sparse_tensor -> Sparse_ops.sparse_tensor -> Config.t -> Sparse_ops.sparse_tensor