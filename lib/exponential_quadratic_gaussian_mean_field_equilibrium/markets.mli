open Types

val create_market_params :
  gamma:float -> k0:float -> k:float -> m0:float -> m:float ->
  sigma0:float -> sigma:float -> v:float -> alpha:float ->
  beta:float -> zeta:float -> eta:float -> market_params

val initialize_market_state : market_params -> market_state

val update_market_state :
  market_state -> ode_solution -> market_params -> float -> market_state

val run_simulation :
  market_params -> agent_state list -> int -> float -> float -> simulation_result

val solve_odes : market_params -> float -> ode_solution