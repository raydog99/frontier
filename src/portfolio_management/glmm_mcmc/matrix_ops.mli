open Torch

module Sparse : sig
  type sparse_matrix
  val to_sparse : Tensor.t -> sparse_matrix
  val from_sparse : sparse_matrix -> Tensor.t
  val sparse_mm : Tensor.t -> Tensor.t -> Tensor.t
end

module Blocked : sig
  val blocked_cholesky : Tensor.t -> Tensor.t
  val blocked_solve : Tensor.t -> Tensor.t -> Tensor.t
end

module Spatial : sig
  val blocked_distance_matrix : Tensor.t -> Tensor.t
  val tapered_matern : Tensor.t -> float -> Tensor.t -> float -> Tensor.t
end