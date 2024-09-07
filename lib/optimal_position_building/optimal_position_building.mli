open Torch

type strategy = float -> float
type multi_asset_strategy = int -> float -> float
type loss_function = float -> float -> float -> float

type market_model = {
  price_process: float -> float -> float;
  volatility: float -> float;
  liquidity: float -> float;
}

type regime = Bull | Bear | Sideways | Volatile

type risk_measure = 
  | ValueAtRisk of float
  | ExpectedShortfall of float
  | MaxDrawdown

type performance_metric = 
  | SharpeRatio
  | SortinoRatio
  | InformationRatio of strategy
  | CalmarRatio

type probability_measure = float -> float

type optimization_constraint =
  | MaxWeight of float
  | MinWeight of float
  | SectorExposure of int array * float * float

type market_impact_model =
  | Linear of float
  | SquareRoot of float
  | PowerLaw of float * float
  | TempermantonPermanent of float * float

type factor = {
  name: string;
  beta: float;
  returns: float array;
}

type objective_function = Tensor.t -> float
type constraint_function = Tensor.t -> bool

type ensemble_method = 
  | EqualWeight
  | InverseVolatility
  | OptimalF
  | KellyWeights

type market_regime = Bull | Bear | Sideways | Volatile

type var_method = Historical | Parametric | MonteCarloVaR
type risk_metric = 
  | ValueAtRisk of float * var_method
  | ConditionalVaR of float * var_method
  | ExpectedShortfall of float
  | DownsideDeviation of float

type simulation_state = {
  time: float;
  prices: float array;
  positions: float array;
  cash: float;
}

type portfolio = {
  weights: float array;
  last_rebalance: float;
}

type backtest_result = {
  returns: float array;
  sharpe_ratio: float;
  max_drawdown: float;
  total_pnl: float;
  total_cost: float;
}

type advanced_backtest_result = {
  base_result: backtest_result;
  var: float;
  cvar: float;
  calmar_ratio: float;
  omega_ratio: float;
  sortino_ratio: float;
}

val risk_neutral_strategy : strategy
val risk_averse_strategy : float -> strategy
val eager_strategy : float -> strategy
val compute_cost : strategy -> strategy -> float -> float -> float -> float

val geometric_brownian_motion : float -> float -> market_model
val jump_diffusion_model : float -> float -> float -> float -> market_model
val stochastic_volatility_model : float -> float -> float -> float -> market_model
val create_regime_switching_model : regime array -> float array array -> (float * float * float) array -> market_model
val simulate_regime_switching_price : market_model -> float -> int -> float array

val best_response_strategy : strategy -> float -> float -> strategy
val two_trader_equilibrium : float -> float -> float -> strategy * strategy
val multi_trader_symmetric_equilibrium : int -> float -> float -> strategy
val n_trader_equilibrium : int -> float -> float -> float -> strategy list
val asymmetric_equilibrium : float list -> float -> float -> float -> strategy list

val calculate_risk : risk_measure -> float array -> float
val apply_risk_constraint : multi_asset_strategy -> risk_measure -> float -> multi_asset_strategy

val calculate_performance : performance_metric -> float array -> float
val rank_strategies : multi_asset_strategy list -> performance_metric -> int -> (int -> market_model) -> float -> float -> float -> int -> (multi_asset_strategy * float) list

val create_lognormal_measure : float -> float -> probability_measure
val expected_cost : strategy -> probability_measure -> float -> float -> float

val particle_swarm_optimization : (float list -> float) -> int -> int -> float list * float
val simulated_annealing : (float list -> float) -> float list -> float -> float -> int -> float list
val genetic_algorithm : (float list -> float) -> int -> int -> int -> float -> float -> float list

val construct_portfolio : multi_asset_strategy -> int -> float array
val rebalance_portfolio : portfolio -> multi_asset_strategy -> float -> float array -> portfolio
val portfolio_optimization : float array -> float array array -> float -> float array
val black_litterman_optimization : 
  float array -> float array array -> float array -> float array array -> 
  float -> float -> optimization_constraint list -> float array
val risk_parity_optimization : 
  float array array -> optimization_constraint list -> float array
val mean_variance_optimization : 
  float array -> float array array -> float -> constraint_function list -> float array
val conditional_value_at_risk_optimization :
  float array -> float array array -> float -> float -> constraint_function list -> float array
val robust_portfolio_optimization :
  float array -> float array array -> float -> float -> constraint_function list -> float array
val risk_budgeting_optimization : float array array -> float array -> float array

val calculate_market_impact : market_impact_model -> float -> float -> float * float

(* Multi-Factor Models *)
val create_factor : string -> float -> float array -> factor
val multi_factor_returns : factor list -> float array -> float array

val calculate_maximum_adverse_excursion : float array -> float
val calculate_maximum_favorable_excursion : float array -> float
val calculate_win_loss_ratio : float array -> float
val calculate_profit_factor : float array -> float
val calculate_beta : float array -> float array -> float
val calculate_alpha : float array -> float array -> float -> float
val calculate_information_ratio : float array -> float array -> float

val calculate_autocorrelation : float array -> int -> float
val calculate_partial_autocorrelation : float array -> int -> float array
val perform_augmented_dickey_fuller_test : float array -> float * float

val plot_strategies : (string * strategy) list -> string -> unit
val plot_drawdown_chart : float array -> string -> unit
val plot_underwater_chart : float array -> string -> unit
val plot_rolling_sharpe_ratio : float array -> int -> string -> unit
val plot_rolling_sortino_ratio : float array -> int -> string -> unit
val plot_rolling_beta : float array -> float array -> int -> string -> unit
val plot_efficient_frontier : multi_asset_strategy list -> int -> (int -> market_model) -> float -> float -> float -> int -> string -> unit
val plot_strategy_comparison : multi_asset_strategy list -> int -> (int -> market_model) -> float -> float -> float -> int -> string -> unit

val combine_strategies : multi_asset_strategy list -> ensemble_method -> multi_asset_strategy
val ensemble_strategy : multi_asset_strategy list -> float array -> multi_asset_strategy
val adaptive_strategy : multi_asset_strategy list -> (float -> float array) -> multi_asset_strategy

(* Adaptive Strategy Optimization *)
val detect_market_regime : float array -> int -> market_regime
val optimize_strategy_for_regime : multi_asset_strategy -> market_regime -> multi_asset_strategy

val optimize_execution_with_transaction_costs : 
  multi_asset_strategy -> market_impact_model -> float -> int -> multi_asset_strategy

val backtest_strategy : 
  multi_asset_strategy -> 
  int -> 
  (int -> market_model) -> 
  market_impact_model ->
  float -> float -> float -> 
  int -> 
  backtest_result

val advanced_backtest_strategy : 
  multi_asset_strategy -> 
  int -> 
  (int -> market_model) -> 
  market_impact_model ->
  float -> float -> float -> 
  int -> 
  advanced_backtest_result

val generate_performance_report : backtest_result -> string -> unit
val compare_strategies_report : (string * backtest_result) list -> string -> unit

val discretize_strategy : strategy -> int -> float array
val interpolate_strategy : float array -> strategy
val strategy_to_tensor : strategy -> int -> Tensor.t
val tensor_to_strategy : Tensor.t -> strategy
val strategy_metrics : strategy -> float -> float -> float -> float * float * float
val strategy_moments : strategy -> float * float * float * float
val strategy_entropy : strategy -> float
val strategy_complexity : strategy -> float
val perform_sensitivity_analysis : multi_asset_strategy -> int -> (int -> market_model) -> float -> float -> float -> int -> (string * float) list
val generate_monte_carlo_paths : market_model -> int -> int -> float array array

(* Machine Learning-based Strategy Improvement *)
module MLStrategyImprovement : sig
  type model

  val create_model : int -> int list -> int -> model
  val train_model : model -> float array array -> float array -> int -> unit
  val predict : model -> float array -> float
  val improve_strategy : multi_asset_strategy -> (int -> float array) -> model -> multi_asset_strategy
end

(* Reinforcement Learning Optimization *)
module DDPG : sig
  type actor
  type critic
  type replay_buffer

  val create_actor : int -> int -> actor
  val create_critic : int -> int -> critic
  val create_replay_buffer : int -> replay_buffer
  val train_ddpg : actor -> critic -> replay_buffer -> (int -> market_model) -> int -> int -> float -> float -> float -> actor
end

val initialize_simulation : int -> float -> simulation_state
val simulation_step : 
  simulation_state -> multi_asset_strategy -> (int -> market_model) -> 
  float -> float -> float -> simulation_state * float
val run_interactive_simulation : 
  int -> float -> multi_asset_strategy -> (int -> market_model) -> 
  float -> float -> float -> int -> simulation_state list