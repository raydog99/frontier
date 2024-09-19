open Torch
open Types

val tensor_to_float_array : Tensor.t -> float array
val float_array_to_tensor : float array -> Tensor.t
val eigen_decomposition : Tensor.t -> eigen_decomposition
val sort_eigenpairs : eigen_decomposition -> eigen_decomposition
val validate_portfolio : portfolio -> unit
val log_message : string -> unit