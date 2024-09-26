open Torch

val compute_eigendecomposition : Tensor.t -> Tensor.t * Tensor.t
val compute_overlaps : Tensor.t -> Tensor.t -> Tensor.t
val stieltjes_transform : Tensor.t -> Tensor.t -> Tensor.t
val compute_eigenvector_localization : Tensor.t -> Tensor.t