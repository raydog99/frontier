open Torch

val create_symmetric_gaussian_matrix : int -> Tensor.t
val create_dyson_brownian_motion : int -> float -> Tensor.t
val truncate_matrix : Tensor.t -> int -> Tensor.t
val create_spiked_matrix : int -> float -> Tensor.t
val create_goe_matrix : int -> Tensor.t
val create_wishart_matrix : int -> int -> Tensor.t