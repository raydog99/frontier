open Torch

type function_with_gradient = {
  f: Tensor.t -> Tensor.t;
  grad_f: Tensor.t -> Tensor.t;
}

type active_subspace = {
  dimension: int;
  eigenvectors: Tensor.t;
  eigenvalues: Tensor.t;
}

type sample_result = {
  points: Tensor.t;
  values: Tensor.t;
  gradients: Tensor.t;
}

type measure = {
  density: Tensor.t -> float;
  support: Tensor.t -> bool;
  dimension: int;
}

type conditional_measure = {
  base_measure: measure;
  condition: Tensor.t -> bool;
  conditional_density: Tensor.t -> Tensor.t -> float;
}

type level_config = {
  samples: int;
  dimension: int;
  poly_degree: int;
}

type mlas_result = {
  active_subspaces: active_subspace list;
  level_approximations: Tensor.t list;
  total_approximation: Tensor.t;
}

module BorelMeasure : sig
  val measure : measure -> Tensor.t -> float
  val null_set : Tensor.t -> bool
  val absolutely_continuous : measure -> measure -> Tensor.t -> bool
  val radon_nikodym_derivative : measure -> measure -> Tensor.t -> float
end

module LebesgueIntegral : sig
  val monte_carlo_integrate : (Tensor.t -> float) -> measure -> int -> float
  val adaptive_integrate : (Tensor.t -> float) -> measure -> float -> float
end

module TensorProduct : sig
  val create : measure -> measure -> measure
  val tensor_product_list : measure list -> measure
end

module ActiveSubspace : sig
  val compute_gradient_outer_product : Tensor.t -> Tensor.t
  val draw_samples : int -> int -> [`Gaussian | `Uniform] -> Tensor.t
  val truncated_svd : Tensor.t -> int -> Tensor.t * Tensor.t * Tensor.t
  val estimate_active_subspace : function_with_gradient -> int -> int -> active_subspace
  val project_points : Tensor.t -> active_subspace -> Tensor.t
end

module PolynomialBasis : sig
  type multi_index = int array
  
  type polynomial_space = {
    dimension: int;
    max_degree: int;
    active_vars: int array;
    inactive_vars: int array;
  }

  module HermiteBasis : sig
    val evaluate_hermite : int -> Tensor.t -> Tensor.t
    val evaluate_multivariate : multi_index -> Tensor.t -> Tensor.t
    val generate_normalized_basis : polynomial_space -> Tensor.t -> Tensor.t list
  end

  module TensorProduct : sig
    val create_space : int array -> int array -> int -> polynomial_space
    val generate_basis : polynomial_space -> Tensor.t -> Tensor.t list
    val project_function : (Tensor.t -> Tensor.t) -> polynomial_space -> Tensor.t -> measure -> (Tensor.t -> Tensor.t)
  end

  module AdaptiveBasis : sig
    type adaptivity_criterion = 
      | LegendreCoeffDecay
      | HermiteCoeffDecay
      | SobolevNorm

    val select_optimal_degree : (Tensor.t -> Tensor.t) -> polynomial_space -> Tensor.t -> measure -> adaptivity_criterion -> float -> int
  end
end

module ErrorAnalysis : sig
  module SVDError : sig
    type svd_error = {
      truncation_error: float;
      reconstruction_error: float;
      condition_number: float;
      rank_selection_error: float;
    }

    val analyze_svd : Tensor.t -> int -> svd_error
  end

  module HierarchicalError : sig
    type error_decomposition = {
      level_errors: float array;
      total_error: float;
      relative_contributions: float array;
    }

    val compute_level_errors : (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> Tensor.t -> int -> float array
    val analyze_error_contributions : float array -> error_decomposition
    val estimate_convergence_rates : float array -> int -> float array
  end
end

module MultilevelAS : sig
  type mlas_config = {
    max_levels: int;
    initial_rank: int;
    max_rank: int;
    tol: float;
    initial_samples: int;
  }

  module LevelManager : sig
    type level_data = {
      active_subspace: active_subspace;
      polynomial_approx: PolynomialBasis.polynomial_space;
      error_estimate: float;
      work_estimate: float;
    }

    val create_level : function_with_gradient -> int -> int -> int -> level_data
    val update_error_estimates : level_data array -> Tensor.t -> level_data array
  end

  module WorkBalance : sig
    val optimize_work_distribution : LevelManager.level_data array -> float -> LevelManager.level_data array
  end

  val run_mlas : mlas_config -> function_with_gradient -> Tensor.t -> mlas_result
end

module Numerics : sig
  val chunk_operation : Tensor.t -> int -> (Tensor.t -> Tensor.t) -> Tensor.t

  module StabilityMonitor : sig
    type stability_status = {
      condition_number: float;
      residual_norm: float;
      gradient_variation: float;
      is_stable: bool;
    }

    val check_stability : Tensor.t -> Tensor.t -> stability_status
    val stabilize_computation : stability_status -> Tensor.t -> Tensor.t -> Tensor.t * bool
  end
end