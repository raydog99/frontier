open Torch

type t
type market_regime = Bull | Bear | Neutral

type economic_factor = {
    name: string;
    value: float;
    impact: Tensor.t;
    forecast_model: (float array -> float) option;
  }

exception Invalid_input of string

val create :
    Tensor.t ->
    Tensor.t ->
    float ->
    Tensor.t ->
    Tensor.t ->
    Tensor.t ->
    Tensor.t ->
    float ->
    int ->
    float ->
    Tensor.t ->
    Tensor.t ->
    (market_regime * market_regime * float) list ->
    (string * float) list ->
    economic_factor list ->
    t

val set_ml_model : t -> (Tensor.t -> Tensor.t) -> t
val set_reinforcement_learning_model : t -> (t -> Tensor.t -> Tensor.t) -> t
val set_liquidity_model : t -> (Tensor.t -> Tensor.t) -> t
val set_network_impact_model : t -> (Tensor.t -> Tensor.t -> Tensor.t) -> t
val set_regime_detection_model : t -> (Tensor.t -> market_regime) -> t
val set_anomaly_detection_model : t -> (Tensor.t -> bool) -> t
val set_deep_rl_model : t -> (module DeepRL.DeepRL) -> t
val set_dynamic_risk_model : t -> (Tensor.t -> float) -> t
val set_market_regime : t -> market_regime -> t

val calculate_tau : t -> float
val calculate_covariance : t -> Tensor.t
val forecast_economic_factors : t -> float -> economic_factor list
val apply_economic_factors : t -> Tensor.t -> Tensor.t
val market_impact_model : t -> Tensor.t -> Tensor.t
val network_impact_model : t -> Tensor.t -> Tensor.t -> Tensor.t
val transaction_cost : t -> Tensor.t -> Tensor.t
val liquidity_adjustment : t -> Tensor.t -> Tensor.t
val detect_regime : t -> Tensor.t -> market_regime
val detect_anomaly : t -> Tensor.t -> bool
val dynamic_risk_assessment : t -> Tensor.t -> float
val deep_rl_optimization : t -> Tensor.t -> Tensor.t
val adaptive_execution : t -> Tensor.t -> Tensor.t array -> Tensor.t array
val generate_market_scenarios : t -> int -> Tensor.t array
val stress_test : t -> Tensor.t -> (string * 'a) list -> (string * Tensor.t * Tensor.t * Tensor.t) list
val multi_period_optimization : t -> Tensor.t list
val parallel_monte_carlo_simulation : t -> Tensor.t -> int -> Tensor.t * Tensor.t * Tensor.t
val multi_asset_optimization : t -> Tensor.t
val calculate_total_cost : t -> Tensor.t -> Tensor.t
val adaptive_execution : t -> Tensor.t -> Tensor.t array -> Tensor.t array