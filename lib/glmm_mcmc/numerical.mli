open Torch

val safe_cholesky : Tensor.t -> Tensor.t
val safe_inverse : Tensor.t -> Tensor.t
val log_sum_exp : Tensor.t -> float
val stable_sigmoid : Tensor.t -> Tensor.t
val log1p_exp : Tensor.t -> Tensor.t