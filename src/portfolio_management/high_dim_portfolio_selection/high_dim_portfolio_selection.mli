open Torch

type utility_type = 
  | Linear of float * float     (** a, η *)
  | Quadratic of float * float  (** η, λ *)
  | Exponential of float * float (** a, η *)
  | Power of float * float      (** γ, η *)
  | Logarithmic of float        (** η *)

type portfolio = {
  weights: Tensor.t;
  utility: utility_type;
  lambda: float;
  cardinality: int option;
}

type market_data = {
  prices: Tensor.t;        (** T x N price matrix *)
  volumes: Tensor.t;       (** T x N volume matrix *)
  market_caps: Tensor.t;   (** N market capitalizations *)
  dates: int array;        (** T dates *)
  index_name: string;      (** Index identifier *)
}

type optimization_config = {
  max_iter: int;
  tol: float;
  screening_freq: int;
  step_size: float;
  min_active_set: int;
}

type performance_metrics = {
  returns: float;
  volatility: float;
  sharpe_ratio: float;
  max_drawdown: float;
  turnover: float;
  active_positions: int;
}

val optimize : 
  ?config:optimization_config ->
  portfolio ->
  market_data ->
  portfolio

val create_kelly_portfolio :
  ?eta:float ->
  ?min_weight:float ->
  market_data ->
  portfolio

val create_hara_portfolio :
  ?gamma:float ->
  ?eta:float ->
  ?risk_aversion:float ->
  market_data ->
  portfolio

val screen_features :
  portfolio ->
  market_data ->
  int list * int list  (** (active_set, screened_set) *)

val evaluate_performance :
  portfolio ->
  market_data ->
  performance_metrics

type validation_result = {
  error_bounds: float * float;  (** Lower and upper bounds *)
  confidence_level: float;
  hypothesis_tests: (string * float * float) list;  (** (test_name, statistic, p_value) *)
}

val validate_portfolio :
  portfolio ->
  market_data ->
  ?confidence_level:float ->
  validation_result

val prepare_market_data :
  prices:Tensor.t ->
  volumes:Tensor.t ->
  market_caps:Tensor.t ->
  dates:int array ->
  index_name:string ->
  market_data

val evaluate_utility : utility_type -> Tensor.t -> Tensor.t
val gradient_utility : utility_type -> Tensor.t -> Tensor.t

type market_stats = {
  correlations: Tensor.t;
  volatilities: Tensor.t;
  mean_correlation: float;
  mean_volatility: float;
  max_drawdown: float;
}

type default_kelly_params = {
  eta: float;
  min_weight: float;
}

type default_hara_params = {
  gamma: float;
  eta: float;
  risk_aversion: flaot;
}

val analyze_market : market_data -> market_stats

val rebalance_portfolio :
  portfolio ->
  market_data ->
  ?transaction_costs:bool ->
  portfolio * performance_metrics

val cross_validate :
  portfolio ->
  market_data ->
  num_folds: int ->
  (performance_metrics list * float * float)  (** (metrics, mean, std) *)

type robustness_result = {
  parameter_sensitivity: (float * performance_metrics) list;
  stability_measure: float;
  turnover_analysis: float list;
}

val check_robustness :
  portfolio ->
  market_data ->
  robustness_result

val compare_strategies :
  portfolio list ->
  market_data ->
  (string * performance_metrics) list  (** (strategy_name, metrics) *)

val default_optimization_config : optimization_config