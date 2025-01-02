open Torch

module IsotropicCore = sig
  val create_correlation_matrix : float -> int -> Tensor.t
  val create_covariance_matrix : float -> float -> int -> Tensor.t
  
  val compute_portfolio_variance : 
    weights:Tensor.t -> 
    cov:Tensor.t -> 
    float

  val homoskedastic_portfolio_variance : 
    sigma:float ->
    rho:float ->
    weights:Tensor.t ->
    {
      total_variance: float;
      systematic_component: float;
      residual_component: float;
      effective_dof: float;
    }

  val compute_factor_decomposition :
    returns:Tensor.t ->
    {
      market_factor: Tensor.t;
      factor_loadings: Tensor.t;
      residual_returns: Tensor.t;
      systematic_risk: float;
      residual_risk: float;
      variance_ratio: float;
    }

  val compute_limiting_behavior :
    sigma:float ->
    rho:float ->
    n:int ->
    {
      asymptotic_variance: float;
      convergence_rate: float;
      min_correlation: float;
      max_effective_dof: float;
    }
end

module MatrixOps = sig
  val compute_grand_sum : Tensor.t -> float
  val compute_trace : Tensor.t -> float
  val tensor_of_ones : int -> Tensor.t
  val diagonal_matrix : Tensor.t -> Tensor.t
  val verify_matrix_properties : 
    Tensor.t -> 
    {
      is_symmetric: bool;
      is_positive_definite: bool;
      has_unit_diagonal: bool;
      valid_correlations: bool;
    }
end

module EmpiricalAnalysis = sig
  val analyze_correlation_distribution :
    returns:Tensor.t ->
    {
      fisher_stats: {
        mean_z: float;
        std_z: float;
        confidence_interval: float * float;
        normality_test: float;
      };
      rank_analysis: {
        effective_rank: float;
        participation_ratio: float;
        top_eigenvalue_ratio: float;
        eigenvalue_spacing: float array;
      };
      isotropy_tests: {
        correlation_variance: float;
        homogeneity_stat: float;
        is_isotropic: bool;
      };
    }

  val compute_effective_dof_curve :
    returns:Tensor.t ->
    max_size:int ->
    samples:int ->
    (int * float) list

  val analyze_normality :
    returns:Tensor.t ->
    {
      shapiro_wilk: float;
      anderson_darling: float;
      ks_test: float * float;
      qq_plot_data: (float * float) array;
      is_normal: bool;
      confidence_level: float;
    }

  val validate_isotropy :
    returns:Tensor.t ->
    {
      correlation_stats: {
        mean: float;
        std: float;
        skewness: float;
        kurtosis: float;
      };
      eigenvalue_stats: {
        bulk_edge: float;
        spectral_gap: float;
        mp_deviation: float;
      };
      stability_metrics: {
        condition_number: float;
        effective_rank: float;
        numerical_stability: bool;
      };
    }
end

module CrossValidation = sig
  val validate_isotropic_model :
    returns:Tensor.t ->
    folds:int ->
    {
      correlation_scores: {
        mean_error: float;
        std_error: float;
        confidence_bounds: float * float;
      };
      rank_scores: {
        mean_effective_rank: float;
        rank_stability: float;
        model_consistency: float;
      };
      model_selection: {
        best_rho: float;
        model_evidence: float;
        cross_validation_error: float;
      };
    }
end

module PortfolioOptimization = sig
  val compute_optimal_weights :
    alpha:Tensor.t ->
    sigma:float ->
    rho:float ->
    lambda:float ->
    n:int ->
    Tensor.t * float  (* weights and expected return *)

  val optimize_with_constraints :
    alpha:Tensor.t ->
    sigma:float ->
    rho:float ->
    constraints:[ 
      | `LongOnly
      | `Sector of (int array * float * float) list
      | `Tracking of (Tensor.t * float)
      | `Turnover of (Tensor.t * float)
    ] list ->
    Tensor.t * float

  val analyze_weight_transition :
    alpha:Tensor.t ->
    sigma:float ->
    rho:float ->
    n_range:int list ->
    {
      weights: (int * Tensor.t) list;
      transition_point: int;
      convergence_rate: float;
      limiting_weights: Tensor.t;
    }

  val compute_risk_contributions :
    weights:Tensor.t ->
    sigma:float ->
    rho:float ->
    float array

  val optimize_hierarchy :
    alpha:Tensor.t ->
    sigma:float ->
    rho:float ->
    hierarchy:(int list) list ->
    Tensor.t
end

module RiskDecomposition = sig
  val portfolio_risk_decomposition :
    weights:Tensor.t ->
    sigma:float ->
    rho:float ->
    n:int ->
    {
      total_risk: float;
      systematic_risk: float;
      residual_risk: float;
      risk_ratio: float;
    }

  val marginal_risk_contributions :
    weights:Tensor.t ->
    cov:Tensor.t ->
    float array

  val component_var :
    weights:Tensor.t ->
    cov:Tensor.t ->
    confidence:float ->
    float array
end

module AsymptoticAnalysis = sig
  val compute_convergence_properties :
    returns:Tensor.t ->
    max_size:int ->
    {
      theoretical_rate: float;
      empirical_rate: float;
      size_effects: (int * float) array;
      stable_region: int * int;
      reliability: float;
    }

  val analyze_limiting_behavior :
    alpha:Tensor.t ->
    sigma:float ->
    rho:float ->
    size_range:int list ->
    {
      small_n: {
        weights: Tensor.t;
        variance: float;
        stability: float;
      };
      transition: {
        start_size: int;
        end_size: int;
        convergence_rate: float;
      };
      large_n: {
        limiting_weights: Tensor.t;
        asymptotic_variance: float;
        numerical_stability: bool;
      };
    }

  val compute_asymptotic_properties :
    rho:float ->
    sizes:int list ->
    {
      theoretical_dof: float array;
      variance_ratios: float array;
      stable_sizes: int list;
    }
end