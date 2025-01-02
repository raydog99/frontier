open Torch
open Types

val calculate_optimal_strategy :
  Tensor.t -> Tensor.t -> market_params -> eqg_solution -> Tensor.t

val update_wealth :
  wealth -> trading_strategy -> risk_premium -> float -> float -> wealth

val update_factor :
  Tensor.t -> market_params -> float -> Tensor.t

val calculate_terminal_liability :
  Tensor.t -> Tensor.t -> market_params -> ode_solution -> eqg_solution -> float

val update_agent_state :
  agent_state -> market_state -> ode_solution -> eqg_solution -> market_params -> float ->
  agent_state * trading_strategy * Tensor.t

val batch_update_agent_states :
  agent_state list -> market_state -> ode_solution -> eqg_solution -> market_params -> float ->
  agent_state list * Tensor.t list * Tensor.t list