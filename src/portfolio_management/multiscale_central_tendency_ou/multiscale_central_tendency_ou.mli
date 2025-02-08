open Torch

type dim3 = {
  x: float;
  y: float;
  z: float;
}

type mctou_params = {
  kappa: float;          (** Mean reversion speed for X1 *)
  epsilon: float;        (** Fast mean reversion parameter *)
  delta: float;          (** Slow mean reversion parameter *)
  alpha2: float;         (** Mean level for X2 *)
  alpha3: float;         (** Mean level for X3 *)
  sigma1: float;         (** Volatility for X1 *)
  sigma2: float;         (** Volatility for X2 *)
  sigma3: float;         (** Volatility for X3 *)
  rho12: float;         (** Correlation between X1 and X2 *)
  rho13: float;         (** Correlation between X1 and X3 *)
  rho23: float;         (** Correlation between X2 and X3 *)
  lambda1: float;       (** Market price of risk for X1 *)
  lambda2: float;       (** Market price of risk for X2 *)
  lambda3: float;       (** Market price of risk for X3 *)
}

type state = {
  x1: Tensor.t;
  x2: Tensor.t;
  x3: Tensor.t;
}

type brownian_motion = {
  dw1: Tensor.t;
  dw2: Tensor.t;
  dw3: Tensor.t;
}

(* Numerical Methods *)
module Numerical : sig
  val adaptive_quadrature : (float -> float) -> float -> float -> float -> float
  val integrate : (float -> float) -> float -> float -> int -> float
  val cholesky_decomposition : Tensor.t -> Tensor.t
end

(* MCTOU Core Functions *)
val create_correlation_matrix : mctou_params -> Tensor.t
val create_volatility_matrix : mctou_params -> Tensor.t
val create_drift_matrix : mctou_params -> Tensor.t
val create_lambda_vector : mctou_params -> Tensor.t
val create_mean_vector : mctou_params -> Tensor.t
val evolve_p : state -> mctou_params -> float -> brownian_motion -> state
val evolve_q : state -> mctou_params -> float -> brownian_motion -> state

(* Futures Pricing *)
module FuturesPricing : sig
  type futures_contract = {
    maturity: float;
    notional: float;
  }

  module Pricing : sig
    val compute_ak : mctou_params -> float -> float -> Tensor.t
    val compute_beta : mctou_params -> float -> float -> float
    val compute_price : mctou_params -> futures_contract -> state -> float -> float
    val compute_price_dynamics : mctou_params -> state -> float -> 
      Tensor.t * Tensor.t
  end
end

(* Portfolio Optimization *)
module Portfolio : sig
  type portfolio_params = {
    gamma: float;         (** Risk aversion parameter *)
    contracts: FuturesPricing.futures_contract array;
    t_horizon: float;     (** Investment horizon *)
  }

  val compute_optimal_strategy : portfolio_params -> state -> float -> Tensor.t
  val simulate_wealth : portfolio_params -> state -> Tensor.t -> 
    float -> float -> brownian_motion -> float
end

(* Portfolio Constraints *)
module PortfolioConstraints : sig
  type constraints = {
    position_limits: (float * float) array;
    total_exposure: float option;
    leverage: float option;
  }

  type transaction_costs = {
    fixed: float;
    proportional: float;
    market_impact: float;
  }

  val projection : Tensor.t -> constraints -> Tensor.t
  val compute_transaction_costs : transaction_costs -> Tensor.t -> 
    Tensor.t -> Tensor.t -> float
end

(* HJB Solution*)
module HJB : sig
  type hjb_params = {
    grid_w: float array;
    nw: int;
    nt: int;
    theta: float;
  }

  module Value_Function : sig
    val compute_value_function : portfolio_params -> state -> float -> float -> float
  end

  module Solver : sig
    val solve : hjb_params -> state -> float -> float -> Tensor.t
  end

  module Verification : sig
    type verification_result = {
      viscosity_subsolution: bool;
      viscosity_supersolution: bool;
      boundary_conditions: bool;
      monotonicity: bool;
      concavity: bool;
    }

    val verify_viscosity_solution : portfolio_params -> state -> float -> 
      float -> Tensor.t -> verification_result
  end
end

(* Risk Management *)
module RiskManagement : sig
  type risk_metrics = {
    var: float;              (** Value at Risk *)
    cvar: float;             (** Conditional Value at Risk *)
    volatility: float;       (** Portfolio volatility *)
    sharpe_ratio: float;     (** Sharpe ratio *)
  }

  type risk_decomposition = {
    systematic: Tensor.t;     (** Systematic risk *)
    idiosyncratic: Tensor.t;  (** Idiosyncratic risk *)
    basis: Tensor.t;          (** Basis risk *)
  }

  val compute_risk_metrics : portfolio_params -> state -> float -> 
    Tensor.t -> int -> risk_metrics
  val decompose_risk : portfolio_params -> state -> Tensor.t -> risk_decomposition
end

(* Market Verification *)
module MarketVerification : sig
  type market_properties = {
    complete: bool;
    spanning_rank: int;
    volatility_rank: int;
    martingale: bool;
  }

  type no_arbitrage_result = {
    no_arbitrage: bool;
    price_consistent: bool;
    martingale_property: bool;
  }

  val verify_market_completion : portfolio_params -> state -> float -> market_properties
  val verify_martingale : portfolio_params -> state -> float -> float -> bool array
  val verify_no_arbitrage : portfolio_params -> float array -> state -> 
    float -> no_arbitrage_result
end

(* Solution Analysis *)
module SolutionAnalysis : sig
  type solution_properties = {
    optimal: bool;
    stable: bool;
    error_bound: float;
    condition_number: float;
  }

  val analyze_numerical_properties : portfolio_params -> state -> float -> 
    Tensor.t -> solution_properties
  val analyze_sensitivity : portfolio_params -> state -> float -> Tensor.t -> 
    Tensor.t * Tensor.t * Tensor.t
end