open Torch

type barrier_params = {
  mu: float;
  beta: float;
  max_inner_iter: int;
  tolerance: float;
}

type constraint_type =
  | Martingale
  | Marginal
  | Support
  | Moment
  | CustomLinear of (Tensor.t -> float)

val solve_interior_point : MOT.t -> barrier_params -> Tensor.t
val solve_entropic : MOT.t -> float -> int -> Tensor.t
val project_constraints : Tensor.t -> constraint_type array -> float -> Tensor.t