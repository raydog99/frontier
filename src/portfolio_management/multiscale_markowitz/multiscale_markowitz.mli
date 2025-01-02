open Torch

type asset_data = {
  returns: Tensor.t;
  scales: float array;
  names: string array;
}

type portfolio_params = {
  target_hurst: float;
  min_weight: float;
  max_weight: float;
}

type scaling_params = {
  beta: float;
  alpha: float;
}

type multifractal_params = {
  q_min: float;
  q_max: float;
  q_steps: int;
  scale_min: float;
  scale_max: float;
  scale_steps: int;
}

type diffusion_params = {
  beta: float;
  alpha: float;
  k_alpha: float;
}

type performance_metrics = {
  sharpe_ratio: float;
  sortino_ratio: float;
  max_drawdown: float;
  annualized_return: float;
  annualized_vol: float;
  information_ratio: float;
}

type backtest_params = {
  lookback_days: int;
  rebalance_frequency: int;
  transaction_cost: float;
  start_date: float;
  end_date: float;
}

val compute_returns : Tensor.t -> Tensor.t
val compute_variance : Tensor.t -> dim:int -> Tensor.t
val compute_covariance : Tensor.t -> Tensor.t -> Tensor.t
val compute_correlation : Tensor.t -> Tensor.t -> Tensor.t
val compute_multiscale_covariance : Types.asset_data -> scales:float array -> Tensor.t
val compute_hurst_exponent : Tensor.t -> scales:float array -> float
val optimize_min_variance : ?constraints:Tensor.t option -> Tensor.t -> Tensor.t
val optimize_max_sharpe : ?constraints:Tensor.t option -> Tensor.t -> Tensor.t -> Tensor.t
val compute_l1_covariance : Tensor.t -> Tensor.t
val compute_robust_multiscale_covariance : Types.asset_data -> scales:float array -> Tensor.t
val solve_fractional_pde : Tensor.t -> Types.diffusion_params -> Tensor.t
val compute_metrics : Tensor.t -> Tensor.t -> Types.performance_metrics
val compute_rolling_returns : Tensor.t -> int -> Tensor.t

module StandardizedHurst : sig
  type hurst_components = {
    beta_n: float array;
    alpha_n: float array;
    h_n: float array;
  }

  val compute_standardized_hurst : Tensor.t -> float array -> hurst_components
  val test_multifractality : hurst_components -> bool
end

module WaveletAnalysis : sig
  type wavelet_params = {
    scales: float array;
    wavelet_type: [`Mexican_hat | `Morlet];
    n_moments: int;
    n_vanishing_moments: int;
  }

  val compute_wtmm : Tensor.t -> wavelet_params -> float array * (float * float * float) list list
end

module RegimeAnalysis : sig
  type regime_state = {
    volatility: float;
    correlation: float;
    hurst: float;
    duration: int;
  }

  val detect_regimes : Tensor.t -> int -> (int * regime_state) list
  val analyze_transitions : (int * regime_state) list -> Tensor.t * float
end

module ScaleCorrelator : sig
  type interaction_effects = {
    correlation_hurst_coupling: float;
    volatility_correlation_feedback: float;
    regime_dependent_scaling: float array;
    cross_scale_correlations: Tensor.t;
  }

  val compute_interactions : Tensor.t -> float array -> (int * RegimeAnalysis.regime_state) list -> interaction_effects
  val adjust_portfolio_weights : Tensor.t -> interaction_effects -> Tensor.t
end