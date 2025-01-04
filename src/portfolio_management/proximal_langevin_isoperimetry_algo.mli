open Torch

type tensor = Tensor.t
type distribution = {
  mean: tensor;
  covariance: tensor;
}

type error =
  | InvalidParameter of string
  | NumericalError of string
  | ConvergenceError of string

type inequality_type =
  | LSI of float      (** α-LSI *)
  | Poincare of float (** β-Poincaré *)

type config = {
  epsilon: float;
  num_steps: int;
  alpha: float option;    (** LSI constant *)
  beta: float option;     (** Poincaré constant *)
  l: float;              (** Gradient smoothness *)
  m: float;              (** Hessian smoothness *)
  dimension: int;
}

type process_result = {
  trajectory: tensor list;
  energies: float list;
  errors: error list;
}

val gradient : (tensor -> float) -> tensor -> tensor
val hessian : (tensor -> float) -> tensor -> tensor
val third_derivative : (tensor -> float) -> tensor -> tensor
val operator_norm : tensor -> tensor
val kl_divergence : tensor -> tensor -> float

val step : config -> tensor -> tensor * error list
val run : config -> (tensor -> float) -> tensor -> process_result

(** Weighted Langevin dynamics *)
module WeightedLangevin : sig
  type weighted_config = {
    base_config: config;
    weight_fn: tensor -> tensor;
    time_factor: float -> float;
  }

  val step : weighted_config -> tensor -> float -> tensor * error list
  val run : weighted_config -> tensor -> process_result
end

(** Analysis *)
module Analysis : sig
  type analysis_result = {
    lsi_constant: float option;
    poincare_constant: float option;
    smoothness_params: (float * float) option;
    error_bounds: float list;
  }

  val verify_lsi : tensor list -> float -> (tensor -> float) -> bool list
  val verify_smoothness : tensor list -> float -> float -> 
                         (tensor -> float) -> bool list
  val analyze_trajectory : config -> (tensor -> float) -> 
                         tensor list -> analysis_result
end

(** Convergence *)
module Convergence : sig
  type convergence_result = {
    rates: (float * float) list;
    theoretical_bounds: float list;
    achieved_accuracy: float;
  }

  val analyze_rates : config -> (tensor -> float) -> 
                     tensor list -> convergence_result
end

(** Stability *)
module Stability : sig
  type stability_result = {
    perturbation_growth: float list;
    mixing_times: float list;
    ergodicity_measure: float;
  }

  val analyze_stability : config -> (tensor -> float) -> 
                        tensor list -> stability_result
end

module Verification : sig
  type verification_result = {
    smoothness_preserved: bool;
    convergence_achieved: bool;
    stability_verified: bool;
    error_summary: error list;
  }

  val verify_all : config -> (tensor -> float) -> 
                  process_result -> verification_result
end