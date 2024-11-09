open Torch

type dimension_spec = {
  spatial_dims: int array;
  time_dims: int;
  batch_size: int option;
}

type solver_config = {
  dimension_spec: dimension_spec;
  regularization: float;
  max_iter: int;
  tolerance: float;
  batch_size: int option;
}

val create_grid : dimension_spec -> (float * float) array -> int -> Tensor.t
val solve : MOT.t -> solver_config -> Tensor.t
val compute_marginals : Tensor.t -> int array -> Tensor.t