open Torch

val stable_inverse_sqrt : Tensor.t -> Tensor.t
val stable_eigendecomposition : Tensor.t -> Tensor.t * Tensor.t
val stable_matrix_multiply : Tensor.t -> Tensor.t -> Tensor.t
val project_to_psd : Tensor.t -> Tensor.t