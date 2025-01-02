open Torch

val mean : float array -> float
val variance : float array -> float
val standard_deviation : float array -> float
val covariance_matrix : float array array -> Tensor.t
val correlation_matrix : float array array -> Tensor.t