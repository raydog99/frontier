open Torch
open Types

type grid_point = {
  lambda: float;
  gamma: float;
  score: float;
  gradient: float * float;
  region_score: float;
}

val compute_theoretical_bounds : Tensor.t -> float * (float * float) array

val create_adaptive_grid : Tensor.t -> int -> grid_point list

val cross_validate : Tensor.t -> model_params -> int -> (float * float) option

val select_parameters : Tensor.t -> int -> model_params