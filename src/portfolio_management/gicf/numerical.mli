open Torch

val safe_cholesky : Tensor.t -> Tensor.t result

val safe_inverse : Tensor.t -> Tensor.t result

val monitor_conditioning : Tensor.t -> float -> Tensor.t result

val safe_logdet : Tensor.t -> float result

val scale_computations : Tensor.t -> Tensor.t * float

val stabilize_matrix : Tensor.t -> float -> Tensor.t