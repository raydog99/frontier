open Torch
open Types

val optimize_subproblem : portfolio -> covariance_matrix -> optimization_params -> optimization_result
val aggregate_solutions : portfolio array -> optimization_result array -> portfolio