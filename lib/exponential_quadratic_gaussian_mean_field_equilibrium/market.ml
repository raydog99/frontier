open Torch
open Types
open Logging

let create_market_params ~gamma ~k0 ~k ~m0 ~m ~sigma0 ~sigma ~v ~alpha ~beta ~zeta ~eta =
  { gamma; k0; k; m0; m; sigma0; sigma; v; alpha; beta; zeta; eta }

let initialize_market_state params =
  let x0 = Tensor.zeros [1] in
  let risk_premium = Risk_premium.initialize_risk_premium params in
  let kalman_state = Kalman_bucy.initialize_kalman_state params in
  { x0; risk_premium; kalman_state }

let update_market_state state ode_solution params dt =
  try
    let x0 = Tensor.(add state.x0
      (sub (mul (float (-1. *. params.k0)) (sub state.x0 (float params.m0)))
           (mul (float params.sigma0) (randn [1] ~mean:(Scalar.float 0.) ~std:(Scalar.float (sqrt dt)))))) in
    let risk_premium = Risk_premium.update_risk_premium state.risk_premium params dt in
    let predicted_kalman_state = Kalman_bucy.predict state.kalman_state params dt in
    let observation = Tensor.(add x0 (mul risk_premium (float dt))) in
    let updated_kalman_state = Kalman_bucy.update predicted_kalman_state observation params dt in
    { x0; risk_premium; kalman_state = updated_kalman_state }
  with
  | Failure msg -> 
    error (Printf.sprintf "Failed to update market state: %s" msg);
    raise (SimulationError msg)

let run_simulation params agents num_steps dt epsilon =
  info "Starting simulation";
  let market_state = initialize_market_state params in
  let ode_solution = Ode_solver.solve_odes params (float_of_int num_steps *. dt) in
  let eqg_solution = Eqg_solver.solve_eqg params ode_solution (float_of_int num_steps *. dt) in
  
  let rec simulate step market_state agents market_clearing_state =
    if step >= num_steps then
      (market_state, agents, market_clearing_state, None)
    else
      let updated_market_state = update_market_state market_state ode_solution params dt in
      let updated_agents, strategies, _ = List.split3 (List.map (fun agent ->
        Agent.update_agent_state agent updated_market_state ode_solution eqg_solution params dt
      ) agents) in
      let updated_market_clearing_state = Market_clearing.update_market_clearing_state strategies in
      let adjusted_risk_premium = Market_clearing.adjust_risk_premium 
        updated_market_state.risk_premium updated_market_clearing_state params in
      let final_market_state = { updated_market_state with risk_premium = adjusted_risk_premium } in
      
      if Market_clearing.is_market_cleared updated_market_clearing_state.average_strategy epsilon then
        (final_market_state, updated_agents, updated_market_clearing_state, Some step)
      else
        simulate (step + 1) final_market_state updated_agents updated_market_clearing_state
  in
  let initial_market_clearing_state = Market_clearing.update_market_clearing_state 
    (List.map (fun _ -> Tensor.zeros [1]) agents) in
  try
    let result = simulate 0 market_state agents initial_market_clearing_state in
    info (Printf.sprintf "Simulation completed in %d steps" (Option.value (snd result) ~default:num_steps));
    { final_market_state = fst result; 
      final_agents = snd result; 
      final_market_clearing_state = fst (snd (snd result));
      convergence_step = snd (snd (snd result)) }
  with
  | Failure msg -> 
    error (Printf.sprintf "Simulation failed: %s" msg);
    raise (SimulationError msg)
  | e -> 
    error (Printf.sprintf "Unexpected error: %s" (Printexc.to_string e));
    raise e

let solve_odes = Ode_solver.solve_odes