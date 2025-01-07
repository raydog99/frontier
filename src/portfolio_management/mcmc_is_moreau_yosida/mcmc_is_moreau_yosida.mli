open Torch

type dimension = int
type scaling_param = float

type sampling_config = {
  n_samples: int;
  n_chains: int;
  n_warmup: int;
  initial_lambda: float;
  adapt_mass: bool;
  adapt_lambda: bool;
  target_accept: float;
}

type sampling_stats = {
  acceptance_rate: float;
  effective_samples: int;
  r_hat: float;
  max_tree_depth: int option;
}

type sampling_result = {
  samples: Tensor.t array;
  stats: sampling_stats;
  tuning: mass_matrix_stats;
  asymptotic: asymptotic_stats;
}

type mass_matrix_stats = {
  condition_number: float;
  scaling_factor: float;
  efficiency: float;
}

type asymptotic_stats = {
  variance: Tensor.t;
  convergence_rate: float;
  normality_test: float;
}

module ConvexFunction : sig
  type t = {
    f: Tensor.t -> Tensor.t;
    domain: Tensor.t -> bool;
    lower_bound: float option;
  }

  val is_proper_lsc_convex : t -> bool
  val eval : t -> Tensor.t -> Tensor.t
end

val envelope : ConvexFunction.t -> float -> Tensor.t -> Tensor.t
val proximal_map : ConvexFunction.t -> float -> Tensor.t -> Tensor.t
val verify_lipschitz : ConvexFunction.t -> float -> Tensor.t -> float -> bool

module MALA : sig
    val step : ConvexFunction.t -> float -> Tensor.t -> Tensor.t * bool
    val run : sampling_config -> ConvexFunction.t -> Tensor.t -> sampling_result
end

module πλ : sig
    val step : ConvexFunction.t -> float -> Tensor.t -> Tensor.t * bool
    val run : sampling_config -> ConvexFunction.t -> Tensor.t -> sampling_result
end

module ImportanceSampling : sig
  type weight_stats = {
    sum_weights: float;
    effective_sample_size: float;
    max_weight: float;
  }

  val compute_weights : ConvexFunction.t -> float -> Tensor.t array -> Tensor.t array
  val estimate_mean : ConvexFunction.t -> float -> Tensor.t array -> Tensor.t * weight_stats
  val estimate_variance : ConvexFunction.t -> float -> Tensor.t array -> Tensor.t
end

module Quantile : sig
  type quantile_estimate = {
    value: float;
    std_error: float;
    confidence_interval: float * float;
  }

  val estimate : Tensor.t array -> Tensor.t array -> float -> quantile_estimate
end

module Diagnostics : sig
  type monitor_stats = {
    r_hat: float;
    ess: float;
    stable: bool;
    stationarity_test: float;
  }

  val monitor_convergence : Tensor.t array array -> monitor_stats
  val verify_convergence : Tensor.t array -> bool * string list
end

module DimensionScaling : sig
  type scaling_result = {
    optimal_lambda: float;
    dimension_factor: float;
    efficiency_estimate: float;
  }

  val compute_optimal_lambda : Tensor.t -> dimension -> scaling_result
  val verify_scaling : Tensor.t array -> bool * float
end

module Ergodicity : sig
  type ergodicity_result = {
    is_ergodic: bool;
    spectral_gap: float option;
    mixing_time: int option;
  }

  val verify_ergodicity : Tensor.t array -> (Tensor.t -> float) -> float -> ergodicity_result
  val verify_conditions : ConvexFunction.t -> float -> Tensor.t -> bool * string list
end