open Torch
open Types
open Logging

let calculate_optimal_strategy z_i0 estimated_risk_premium params eqg_solution =
  let { phi; psi; chi } = eqg_solution in
  Tensor.(div (add z_i0 estimated_risk_premium) (add (float params.gamma) phi))

let update_wealth wealth strategy risk_premium sigma dt =
  let dW = Tensor.randn [1] ~mean:(Scalar.float 0.) ~std:(Scalar.float (sqrt dt)) in
  wealth +. Tensor.(to_float0_exn (mul (mul strategy (add risk_premium (mul sigma dW))) (float dt)))

let update_factor factor params dt =
  let dW = Tensor.randn [1] ~mean:(Scalar.float 0.) ~std:(Scalar.float (sqrt dt)) in
  Tensor.(add factor
    (sub (mul (float (-1. *. params.k)) (sub factor (float params.m)))
         (mul (float params.sigma) dW)))

let calculate_terminal_liability x0 xi params ode_solution eqg_solution =
  let value_function = Eqg_solver.calculate_value_function eqg_solution x0 xi in
  Tensor.(to_float0_exn (neg (log (neg value_function))))

let update_agent_state agent_state market_state ode_solution eqg_solution params dt =
  try
    let estimated_risk_premium = Risk_premium.estimate_risk_premium market_state.kalman_state in
    let z_i0 = Tensor.(mul (transpose (get ode_solution.a10 [|0|]))
      (add agent_state.factor (get ode_solution.b1 [|0|]))) in
    let optimal_strategy = calculate_optimal_strategy z_i0 estimated_risk_premium params eqg_solution in
    let wealth = update_wealth agent_state.wealth optimal_strategy market_state.risk_premium params.sigma dt in
    let factor = update_factor agent_state.factor params dt in
    let value_function = Eqg_solver.calculate_value_function eqg_solution market_state.x0 factor in
    { wealth; factor }, optimal_strategy, value_function
  with
  | Failure msg -> 
    error (Printf.sprintf "Failed to update agent state: %s" msg);
    raise (SimulationError msg)

let batch_update_agent_states agent_states market_state ode_solution eqg_solution params dt =
  let factors = Tensor.stack (List.map (fun agent -> agent.factor) agent_states) ~dim:0 in
  let z_i0 = Tensor.(matmul (get ode_solution.a10 [|0|]) (add factors (get ode_solution.b1 [|0|]))) in
  let estimated_risk_premium = Risk_premium.estimate_risk_premium market_state.kalman_state in
  let optimal_strategies = calculate_optimal_strategy z_i0 estimated_risk_premium params eqg_solution in
  let new_factors = update_factor factors params dt in
  let new_wealths = Tensor.map2 (fun wealth strategy -> 
    update_wealth wealth strategy market_state.risk_premium params.sigma dt
  ) (Tensor.of_float1 (List.map (fun agent -> agent.wealth) agent_states)) optimal_strategies in
  let value_functions = Eqg_solver.calculate_value_function eqg_solution market_state.x0 new_factors in
  List.map2 (fun wealth factor -> 
    { wealth = Tensor.to_float0_exn wealth; factor = Tensor.select factor 0 0 }
  ) (Tensor.to_list1 new_wealths) (Tensor.unbind new_factors ~dim:0),
  Tensor.unbind optimal_strategies ~dim:0,
  Tensor.unbind value_functions ~dim:0