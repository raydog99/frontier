open Torch

val wigner_semicircle_density : float -> Tensor.t -> Tensor.t
val wigner_semicircle_stieltjes : float -> Tensor.t -> Tensor.t
val marcenko_pastur_density : float -> float -> Tensor.t -> Tensor.t
val compute_dos : Tensor.t -> int -> float -> float -> Tensor.t