open Torch

type predictor = {
  epsilon: float;  (* Mean reversion rate *)
  psi: float;     (* Volatility *)
  beta: float;    (* Market beta *)
  max_pos: float; (* Maximum position *)
  gamma: float;   (* Individual cost parameter *)
}

type trading_params = {
  dt: float;                (* Time step *)
  gamma: float;             (* Global cost parameter *)
  lambda: float;            (* Risk aversion *)
  num_assets: int;          (* Number of assets *)
  max_risk: float;          (* Maximum allowed risk *)
  max_cost: float;          (* Maximum allowed cost *)
  initial_capital: float;   (* Initial capital *)
  predictors: predictor array;  (* Array of predictors *)
}

type position = {
  value: float;
  asset_idx: int;
}

(* Numerical utilities *)
val adaptive_integrate : ?tol:float -> ?max_steps:int -> (float -> float) -> float -> float -> float
val stabilize_covariance : float array array -> float array array

(* Ornstein-Uhlenbeck process *)
module OrnsteinUhlenbeck : sig
  type state = {
    value: Tensor.t;
    time: float;
    path_max: float;
    path_min: float;
    crossings: int;
  }

  type regime = 
    | Weak of float
    | Intermediate of float
    | Strong of float

  val create : float -> state
  val identify_regime : predictor -> trading_params -> regime
  val update_exact : predictor -> state -> float -> state
  val simulate_path : ?n_steps:int -> ?dt:float -> predictor -> float -> state array
  val stationary_density : predictor -> float -> float
end

(* Risk calculation and management *)
module RiskCalculation : sig
  type risk_decomposition = {
    total_risk: float;
    systematic_risk: float;
    specific_risk: float;
    risk_contributions: float array;
  }

  val calculate_risk_decomposition : float array -> predictor array -> risk_decomposition
  val calculate_min_risk : predictor array -> float
end

(* Trading threshold and cost calculations *)
val calculate_threshold : predictor -> trading_params -> float
val calculate_modified_threshold : predictor -> float -> float -> float
val calculate_trading_rate : predictor -> float -> float
val calculate_costs : trading_params -> float array -> float array -> float
val optimize_trade : trading_params -> predictor -> float -> float -> float
val optimize_positions : trading_params -> predictor array -> float array -> float array

(* Trading system *)
module TradingSystem : sig
  type system_state = {
    positions: float array;
    predictors: float array;
    risk: float;
    costs: float;
    pnl: float;
    time: float;
    regime_states: OrnsteinUhlenbeck.regime array;
  }

  val create_system : trading_params -> system_state
  val update_system : trading_params -> system_state -> system_state
  val run_simulation : trading_params -> system_state -> int -> system_state array
end

(* Mean field dynamics *)
module MeanFieldDynamics : sig
  type field_state = {
    positions: float array;
    predictors: float array;
    risk: float;
    trading_rates: float array;
  }

  val create_state : int -> field_state
  val calculate_trading_rate : predictor -> float -> float
  val update_dynamics : trading_params -> field_state -> float -> field_state
end