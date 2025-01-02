open Torch
open Types

val preprocess_correlation_matrix : correlation_matrix -> int -> int -> correlation_matrix
val compute_covariance_matrix : Tensor.t -> covariance_matrix
val compute_correlation_matrix : covariance_matrix -> correlation_matrix