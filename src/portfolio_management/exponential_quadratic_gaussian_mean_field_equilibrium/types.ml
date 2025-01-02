open Torch

type risk_premium = Tensor.t
type stock_price = Tensor.t
type wealth = float
type liability = float
type trading_strategy = Tensor.t

type market_params = {
  gamma: float;
  k0: float;
  k: float;
  m0: float;
  m: float;
  sigma0: float;
  sigma: float;
  v: float;
  alpha: float;
  beta: float;
  zeta: float;
  eta: float;
}

type agent_params = {
  initial_wealth: float;
  initial_factor: float;
}

type ode_solution = {
  a00: Tensor.t;
  a11: Tensor.t;
  a10: Tensor.t;
  b0: Tensor.t;
  b1: Tensor.t;
  c: Tensor.t;
}

type market_state = {
  x0: Tensor.t;
  risk_premium: risk_premium;
  kalman_state: kalman_state;
}

type agent_state = {
  wealth: wealth;
  factor: Tensor.t;
}

type kalman_state = {
  estimate: Tensor.t;
  error_covariance: Tensor.t;
}

type eqg_solution = {
  phi: Tensor.t;
  psi: Tensor.t;
  chi: Tensor.t;
}

type market_clearing_state = {
  average_strategy: Tensor.t;
  clearing_error: float;
}

type simulation_result = {
  final_market_state: market_state;
  final_agents: agent_state list;
  final_market_clearing_state: market_clearing_state;
  convergence_step: int option;
}

exception SimulationError of string