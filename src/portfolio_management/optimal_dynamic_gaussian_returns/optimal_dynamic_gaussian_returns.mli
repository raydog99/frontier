open Torch

(** Core types *)
type signal = {
  data: Tensor.t;
  mean: float;
  std: float;
}

type returns = {
  data: Tensor.t;
  mean: float;
  std: float;
}

type strategy = {
  signals: signal;
  returns: returns;
  correlation: float;
}

type strategy_params = {
  window: int option;
  alpha: float option;
  regularization: float option;
  lookback: int;
}

type cost_model = {
  fixed_cost: float;
  proportional_cost: float;
  quadratic_cost: float;
}

(** Linear Gaussian module *)
module LinearGaussian : sig
  type dimensionless_stats = {
    sharpe: float;
    skewness: float;
    kurtosis: float;
  }

  val calculate_moments : float -> dimensionless_stats
  val maximal_stats : dimensionless_stats
end

(** Total Least Squares optimization *)
module TLS : sig
  type tls_result = {
    beta: Tensor.t;
    residuals: Tensor.t;
    correlation: float;
    degrees_of_freedom: float;
  }

  val solve_regularized : Tensor.t -> Tensor.t -> float -> tls_result
  val cross_validate : Tensor.t -> Tensor.t -> int -> float list -> float
end

(** Standard error calculations *)
module StandardErrors : sig
  type standard_errors = {
    sharpe_stderr: float;
    skewness_stderr: float;
    kurtosis_stderr: float;
  }

  val calc_implied_stderrs : strategy -> int -> standard_errors
  val finite_sample_correction : float -> int -> float
  val confidence_intervals : strategy -> int -> float -> 
    (float * float) * (float * float) * (float * float)
end

(** Portfolio analysis *)
module Portfolio : sig
  type portfolio_stats = {
    sharpe: float;
    skewness: float;
    kurtosis: float;
    correlation_matrix: Tensor.t;
  }

  module CCA : sig
    type cca_result = {
      correlations: Tensor.t;
      signal_weights: Tensor.t;
      return_weights: Tensor.t;
    }

    val compute_correlations : Tensor.t -> Tensor.t -> cca_result
  end

  module MultiAsset : sig
    val mgf_n_assets : int -> float -> float -> float
    val n_asset_moments : int -> float -> float array
    val n_asset_maximal_stats : int -> LinearGaussian.dimensionless_stats
  end

  module Optimization : sig
    val risk_decomposition : strategy array -> float array -> float array
    val risk_parity_weights : strategy array -> float array
  end
end

(** Strategy creation and analysis *)
module Creation : sig
  val create_signal : Tensor.t -> strategy_params -> Tensor.t
  val create_strategy : Tensor.t -> strategy_params -> strategy
end

(** Strategy analysis *)
module Analysis : sig
  type analysis_result = {
    moments: LinearGaussian.dimensionless_stats;
    standard_errors: StandardErrors.standard_errors;
    relative_performance: float * float * float;  (* vs maximal *)
    confidence_intervals: (float * float) * (float * float) * (float * float);
  }

  val analyze_strategy : strategy -> int -> float -> analysis_result
  val risk_decomposition : strategy array -> float array -> float array
end

(** Portfolio construction *)
module PortfolioConstruction : sig
  type portfolio_result = {
    weights: float array;
    statistics: Portfolio.portfolio_stats;
    risk_contributions: float array;
  }

  val create_optimal_portfolio : strategy array -> portfolio_result
end

(** Transaction cost analysis *)
module CostAnalysis : sig
  val analyze_with_costs : strategy -> float array -> cost_model -> 
    float * float * float * int
  val estimate_capacity : strategy -> float -> float
end

(** Extensions and applications *)
module Extensions : sig
  module TransactionCosts : sig
    val calculate_costs : float array -> cost_model -> float
    val optimize_with_costs : strategy -> cost_model -> float -> float * float
  end

  module MultiPeriod : sig
    val autocorrelation : strategy -> int -> float array
    val multi_period_moments : strategy -> int -> 
      {mean: float; variance: float; skewness: float; kurtosis: float; 
       raw_moments: float array}
    val effective_observations : strategy -> int -> float
  end

  module Applications : sig
    val required_sample_size : float -> float -> float -> int option
    val estimate_capacity : strategy -> float -> float
  end
end