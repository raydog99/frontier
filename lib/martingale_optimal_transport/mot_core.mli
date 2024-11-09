open Torch

module type Measure = sig
  type t
  val create : ?support:Tensor.t -> Tensor.t -> t
  val support : t -> Tensor.t
  val density : t -> Tensor.t option
  val sample : t -> int -> Tensor.t
  val expectation : t -> (Tensor.t -> Tensor.t) -> Tensor.t
  val wasserstein_distance : t -> t -> float
end

module type MOT = sig
  type t = {
    marginals: DiscreteMeasure.t array;
    cost: Tensor.t -> Tensor.t -> Tensor.t;
    epsilon: float;
  }

  val create : DiscreteMeasure.t array -> (Tensor.t -> Tensor.t -> Tensor.t) -> float -> t
  val evaluate : t -> Tensor.t -> float
  val get_marginals : t -> DiscreteMeasure.t array
  val get_dimension : t -> int
end

module type Solver = sig
  type solver_params = {
    max_iter: int;
    learning_rate: float;
    tolerance: float;
    discretization_points: int;
  }

  val default_params : solver_params
  val solve : ?params:solver_params -> MOT.t -> Tensor.t
  val project_constraints : MOT.t -> Tensor.t -> Tensor.t
end

type error_metrics = {
  constraint_violation: float;
  objective_gap: float;
  stability_measure: float;
  condition_number: float;
}

type convergence_constants = {
  lipschitz: float;
  moment_bound: float;
  support_bound: float option;
}

val compute_error_metrics : MOT.t -> Tensor.t -> error_metrics
val analyze_stability : MOT.t -> float array -> (float * error_metrics) array
val estimate_error_bounds : MOT.t -> Tensor.t -> float * (float * float)
val compute_convergence_rate : convergence_constants -> int -> float -> float