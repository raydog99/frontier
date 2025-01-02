open Torch

val qr_decomposition : Tensor.t -> Tensor.t * Tensor.t
val compute_residuals : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val sample_matrix : Tensor.t -> Tensor.t -> int -> Tensor.t
val condition_number : Tensor.t -> Tensor.t
val batch_matmul : Tensor.t -> Tensor.t -> int -> Tensor.t
val solve_regularized_ls : Tensor.t -> Tensor.t -> float -> Tensor.t
val parallel_matmul : Tensor.t -> Tensor.t -> int -> Tensor.t
val memory_efficient_qr : Tensor.t -> int -> Tensor.t * Tensor.t