open Torch
open Types

val spectral_clustering : correlation_matrix -> int -> community array
val decompose_portfolio : portfolio -> community array -> portfolio array