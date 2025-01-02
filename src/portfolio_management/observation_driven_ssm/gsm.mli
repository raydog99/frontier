open Torch

type state = {
  a: float;
  b: float;
}

type observation = {
  y: float;  (** Response value *)
  v: float;  (** Exposure *)
  mu: float; (** Known severity parameter *)
}

module StateSpace : sig
  type t = state

  val init : a:float -> b:float -> t
  val validate : t -> bool
  val mean : t -> float
  val variance : t -> float
  val update : t -> observation -> t
end

module BetaPrime : sig
    type params = {
      alpha: float;
      beta: float;
      scale: float;
    }

    val pdf : params:params -> x:float -> float
    val log_pdf : params:params -> x:float -> float
    val cdf : params:params -> x:float -> float
    val generate : params:params -> rng:Random.State.t -> float
end

module PearsonVI : sig
    type params = {
      a: float;
      b: float;
      scale: float;
    }

    val from_beta_prime : BetaPrime.params -> params
    val pdf : params:params -> x:float -> float
    val log_pdf : params:params -> x:float -> float
    val mgf : params:params -> t:float -> float
    val moment : params:params -> k:int -> float
end

module StatisticalValidation : sig
  type validation_result = {
    valid: bool;
    error_msg: string option;
    test_statistics: float array;
  }

  val validate_parameters : 
    SmithMillerComplete.model_params -> validation_result
  
  val validate_state_sequence : 
    state array -> validation_result
  
  val validate_observations : 
    observation array -> validation_result
  
  val validate_model_fit : 
    SmithMillerComplete.model_params -> 
    observation array -> 
    validation_result
end

  module TimeSeriesAnalysis : sig
    val compute_acf : float array -> int -> float array
    val ljung_box_test : float array -> int -> float * float
  end

  module DistributionTests : sig
    val ks_test : float array -> (float -> float) -> bool
    val anderson_darling_test : 
      float array -> (float -> float) -> float * float
  end

  module RobustnessAnalysis : sig
    type robustness_metrics = {
      influence_measures: float array;
      leverage_points: int array;
      cook_distances: float array;
    }

    val compute_influence_measures : 
      SmithMillerComplete.model_params -> 
      observation array -> 
      robustness_metrics
  end

module EdgeCases : sig
  type edge_case_type =
    | ZeroExposure
    | InfiniteMean
    | ZeroVariance
    | DegenerateState
    | BoundaryCondition

  val detect_edge_cases : state -> observation -> edge_case_type option
  val handle_edge_case : edge_case_type -> state -> observation -> state
  val validate_edge_case_handling : state array -> bool
end

module SmithMillerComplete : sig
  type model_params = {
    psi: float;
    a_init: float;
    gamma: float;
  }

  module Init : sig
    val validate_init_params : 
      a_init:float -> psi:float -> gamma:float -> model_params
    val init_state : model_params -> state
  end

  module Observation : sig
    type individual_claim = {
      z: float;
      theta: float;
    }

    val process_observation : 
      model_params -> state -> observation -> float * state
    val conditional_moments : 
      observation -> float -> model_params -> float * float
  end

  module Variance : sig
    val verify_variance_behavior : state array -> float -> bool
    val compute_variance_ratios : state array -> float array
    val analyze_variance_stability : state array -> float -> float array
  end

  val create : psi:float -> a_init:float -> gamma:float -> model_params
  val filter : model_params -> observation array -> (float * state) array
end

module GeneralizedSM : sig
  type model_params = {
    psi: float;
    a_init: float;
    xi: float array;
    convex_space: bool;
  }

  module ParameterSpace : sig
    type xi_params = {
      values: float array;
      bounds: (float * float) array;
    }

    val validate_xi : xi_params -> xi_params
    val create_convex_space : xi_values:float array -> xi_params
    val verify_convexity : xi_params -> bool
  end

  module Functions : sig
    val compute_a : xi:ParameterSpace.xi_params -> state:state -> float
    val compute_b : xi:ParameterSpace.xi_params -> state:state -> float
    val verify_measurable_functions : 
      xi:ParameterSpace.xi_params -> 
      state_seq:state array -> 
      bool
  end

  module WellDefinedness : sig
    type validation_result = {
      is_valid: bool;
      error_msg: string option;
    }

    val validate_components : 
      SmithMillerComplete.model_params -> state -> validation_result
    val validate_updates : 
      SmithMillerComplete.model_params -> 
      observation array -> 
      validation_result
    val check_well_defined : 
      SmithMillerComplete.model_params -> 
      observation array -> 
      validation_result
  end

  val create : 
    psi:float -> 
    a_init:float -> 
    xi:ParameterSpace.xi_params -> 
    require_thinning:bool -> 
    model_params
  
  val update : model_params -> state -> observation -> state
end

module AdaptiveSSM : sig
  type model_params = {
    psi: float;
    a_init: float;
    p_seq: float array;
    q_seq: float array;
  }

  module ParameterValidation : sig
    type validation_result = {
      valid: bool;
      error_msg: string option;
    }

    val validate_p_sequence : float array -> validation_result
    val validate_q_sequence : float array -> validation_result
    val check_parameter_compatibility : 
      float array -> float array -> validation_result
  end

  module StateRecursion : sig
    val update_state : 
      model_params -> state -> int -> observation -> state
    val check_stability : model_params -> state array -> bool
  end

  module Inference : sig
    val forward_filter : 
      model_params -> 
      observation array -> 
      state array * float array
    val backward_smooth : 
      model_params -> 
      state array -> 
      observation array -> 
      state array
  end

  module StateAnalysis : sig
    type state_dynamics = {
      mean_evolution: float array;
      variance_evolution: float array;
      stability_metric: float;
    }

    val analyze_dynamics : state array -> state_dynamics
  end

  module ModelComparison : sig
    type model_fit = {
      aic: float;
      bic: float;
      dic: float;
    }

    val compute_information_criteria : 
      model_params -> 
      observation array -> 
      float array -> 
      model_fit
  end

  module Diagnostics : sig
    type residual_analysis = {
      pearson_residuals: float array;
      deviance_residuals: float array;
      acf: float array;
      qq_points: (float * float) array;
    }

    val analyze_residuals : 
      model_params -> 
      observation array -> 
      state array -> 
      residual_analysis
    val validate_model : 
      model_params -> 
      observation array -> 
      bool * residual_analysis
  end

  val create : 
    psi:float -> 
    a_init:float -> 
    p_seq:float array -> 
    q_seq:float array -> 
    model_params
end