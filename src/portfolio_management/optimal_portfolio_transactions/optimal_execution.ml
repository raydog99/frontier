open Torch
open Lwt

type market_regime = Bull | Bear | Neutral

type economic_factor = {
  name: string;
  value: float;
  impact: Tensor.t;
  forecast_model: (float array -> float) option;
}

type t = {
  initial_portfolio: Tensor.t;
  final_portfolio: Tensor.t;
  time_horizon: float;
  permanent_impact: Tensor.t;
  temporary_impact: Tensor.t;
  volatility: Tensor.t;
  correlation: Tensor.t;
  risk_aversion: float;
  num_periods: int;
  transaction_cost: float;
  market_volume: Tensor.t;
  bid_ask_spread: Tensor.t;
  ml_model: (Tensor.t -> Tensor.t) option;
  current_regime: market_regime;
  regime_transition_probs: (market_regime * market_regime * float) list;
  reinforcement_learning_model: (t -> Tensor.t -> Tensor.t) option;
  risk_limits: (string * float) list;
  liquidity_model: (Tensor.t -> Tensor.t) option;
  network_impact_model: (Tensor.t -> Tensor.t -> Tensor.t) option;
  regime_detection_model: (Tensor.t -> market_regime) option;
  anomaly_detection_model: (Tensor.t -> bool) option;
  economic_factors: economic_factor list;
  deep_rl_model: (module DeepRL.DeepRL) option;
  dynamic_risk_model: (Tensor.t -> float) option;
}

exception Invalid_input of string

let validate_input initial_portfolio final_portfolio time_horizon permanent_impact temporary_impact volatility correlation risk_aversion num_periods transaction_cost market_volume bid_ask_spread regime_transition_probs risk_limits economic_factors =
  let num_assets = Tensor.size initial_portfolio 0 in
  if Tensor.numel initial_portfolio <> Tensor.numel final_portfolio then
    raise (Invalid_input "Initial and final portfolio sizes must match");
  if time_horizon <= 0. then
    raise (Invalid_input "Time horizon must be positive");
  if Tensor.size permanent_impact 0 <> num_assets || Tensor.size permanent_impact 1 <> num_assets then
    raise (Invalid_input "Permanent impact matrix dimensions must match number of assets");
  if Tensor.size temporary_impact 0 <> num_assets || Tensor.size temporary_impact 1 <> num_assets then
    raise (Invalid_input "Temporary impact matrix dimensions must match number of assets");
  if Tensor.size volatility 0 <> num_assets then
    raise (Invalid_input "Volatility vector length must match number of assets");
  if Tensor.size correlation 0 <> num_assets || Tensor.size correlation 1 <> num_assets then
    raise (Invalid_input "Correlation matrix dimensions must match number of assets");
  if risk_aversion < 0. then
    raise (Invalid_input "Risk aversion must be non-negative");
  if num_periods <= 0 then
    raise (Invalid_input "Number of periods must be positive");
  if transaction_cost < 0. then
    raise (Invalid_input "Transaction cost must be non-negative");
  if Tensor.size market_volume 0 <> num_periods || Tensor.size market_volume 1 <> num_assets then
    raise (Invalid_input "Market volume dimensions must match number of periods and assets");
  if Tensor.size bid_ask_spread 0 <> num_assets then
    raise (Invalid_input "Bid-ask spread vector length must match number of assets");
  if List.length regime_transition_probs <> 9 then
    raise (Invalid_input "Regime transition probabilities must be provided for all 9 possible transitions");
  if List.length risk_limits = 0 then
    raise (Invalid_input "At least one risk limit must be specified")

let create initial_portfolio final_portfolio time_horizon permanent_impact temporary_impact volatility correlation risk_aversion num_periods transaction_cost market_volume bid_ask_spread regime_transition_probs risk_limits economic_factors =
  validate_input initial_portfolio final_portfolio time_horizon permanent_impact temporary_impact volatility correlation risk_aversion num_periods transaction_cost market_volume bid_ask_spread regime_transition_probs risk_limits economic_factors;
  { initial_portfolio; final_portfolio; time_horizon; permanent_impact; temporary_impact; volatility; correlation; risk_aversion; num_periods; transaction_cost; market_volume; bid_ask_spread; ml_model = None; current_regime = Neutral; regime_transition_probs; reinforcement_learning_model = None; risk_limits; liquidity_model = None; network_impact_model = None; regime_detection_model = None; anomaly_detection_model = None; economic_factors; deep_rl_model = None; dynamic_risk_model = None }

let set_ml_model t model = { t with ml_model = Some model }
let set_reinforcement_learning_model t model = { t with reinforcement_learning_model = Some model }
let set_liquidity_model t model = { t with liquidity_model = Some model }
let set_network_impact_model t model = { t with network_impact_model = Some model }
let set_regime_detection_model t model = { t with regime_detection_model = Some model }
let set_anomaly_detection_model t model = { t with anomaly_detection_model = Some model }
let set_deep_rl_model t model = { t with deep_rl_model = Some model }
let set_dynamic_risk_model t model = { t with dynamic_risk_model = Some model }
let set_market_regime t regime = { t with current_regime = regime }

let calculate_tau t = t.time_horizon /. (float_of_int t.num_periods)

let calculate_covariance t =
  let diag_vol = Tensor.diag t.volatility in
  Tensor.matmul (Tensor.matmul diag_vol t.correlation) diag_vol

let forecast_economic_factors t horizon =
  List.map (fun factor ->
    match factor.forecast_model with
    | Some model ->
        let historical_data = [| factor.value |] 
        let forecasted_value = model historical_data in
        { factor with value = forecasted_value }
    | None -> factor
  ) t.economic_factors

let apply_economic_factors t tensor =
  List.fold_left (fun acc factor ->
    Tensor.add acc (Tensor.mul (Tensor.of_float factor.value) factor.impact)
  ) tensor t.economic_factors

let market_impact_model t trading_rate =
  let base_impact = 
    match t.ml_model with
    | Some model -> model trading_rate
    | None ->
        let permanent = Tensor.matmul t.permanent_impact trading_rate in
        let temporary = Tensor.matmul t.temporary_impact trading_rate in
        let volume_ratio = Tensor.div trading_rate t.market_volume in
        let nonlinear_impact = Tensor.pow volume_ratio (Tensor.of_float 0.5) in
        Tensor.add permanent (Tensor.mul temporary nonlinear_impact)
  in
  let regime_adjusted_impact = 
    match t.current_regime with
    | Bull -> Tensor.mul base_impact (Tensor.of_float 0.8)
    | Bear -> Tensor.mul base_impact (Tensor.of_float 1.2)
    | Neutral -> base_impact
  in
  apply_economic_factors t regime_adjusted_impact

let network_impact_model t trading_rate portfolio =
  match t.network_impact_model with
  | Some model -> model trading_rate portfolio
  | None -> trading_rate

let transaction_cost t trading_rate =
  let fixed_cost = Tensor.full_like trading_rate t.transaction_cost in
  let variable_cost = Tensor.mul (Tensor.abs trading_rate) (Tensor.mul t.bid_ask_spread (Tensor.of_float 0.5)) in
  Tensor.add fixed_cost variable_cost

let liquidity_adjustment t trading_rate =
  match t.liquidity_model with
  | Some model -> model trading_rate
  | None -> trading_rate

let detect_regime t market_data =
  match t.regime_detection_model with
  | Some model -> model market_data
  | None -> t.current_regime

let detect_anomaly t market_data =
  match t.anomaly_detection_model with
  | Some model -> model market_data
  | None -> false

let dynamic_risk_assessment t market_data =
  match t.dynamic_risk_model with
  | Some model -> model market_data
  | None -> t.risk_aversion

let deep_rl_optimization t state =
  match t.deep_rl_model with
  | Some (module DRL) ->
      let drl_state = DRL.init () in
      let trained_state = DRL.train t 1000 drl_state in
      DRL.select_action trained_state state
  | None -> failwith "Deep RL model not set"

let adaptive_execution t initial_trajectory market_scenarios =
  let num_periods = t.num_periods in
  let num_assets = Tensor.size t.initial_portfolio 0 in
  let num_scenarios = Array.length market_scenarios in
  
  let execute_step t_step holdings scenario_step =
    let market_data = Tensor.cat [holdings; scenario_step] ~dim:0 in
    let detected_regime = detect_regime t market_data in
    let is_anomaly = detect_anomaly t market_data in
    let dynamic_risk = dynamic_risk_assessment t market_data in
    let mut_t = { t with current_regime = detected_regime; risk_aversion = dynamic_risk; initial_portfolio = holdings } in
    
    let optimal_trajectory = 
      if is_anomaly then
        deep_rl_optimization mut_t market_data
      else
        multi_asset_optimization mut_t
    in
    
    Tensor.get optimal_trajectory [1]  (* Return the next step's holdings *)
  in
  
  let execute_scenario scenario =
    let trajectory = Tensor.zeros [num_periods + 1; num_assets] in
    Tensor.copy_ (Tensor.get trajectory [0]) t.initial_portfolio;
    
    for i = 1 to num_periods do
      let prev_holdings = Tensor.get trajectory [i-1] in
      let scenario_step = Tensor.get scenario [i-1] in
      let new_holdings = execute_step (float_of_int i *. calculate_tau t) prev_holdings scenario_step in
      Tensor.copy_ (Tensor.get trajectory [i]) new_holdings
    done;
    
    trajectory
  in
  
  Array.map execute_scenario market_scenarios


let generate_market_scenarios t num_scenarios =
  let num_periods = t.num_periods in
  let num_assets = Tensor.size t.initial_portfolio 0 in
  
  let generate_scenario () =
    let scenario = Tensor.zeros [num_periods; num_assets] in
    let cov = calculate_covariance t in
    let cholesky = Tensor.cholesky cov in
    
    for i = 0 to num_periods - 1 do
      let random_shock = Tensor.randn [1; num_assets] in
      let correlated_shock = Tensor.matmul random_shock cholesky in
      let forecasted_factors = forecast_economic_factors t (float_of_int i *. calculate_tau t) in
      let economic_shock = apply_economic_factors { t with economic_factors = forecasted_factors } correlated_shock in
      Tensor.copy_ (Tensor.get scenario [i]) economic_shock
    done;
    
    scenario
  in
  
  Array.init num_scenarios (fun _ -> generate_scenario ())

let stress_test t trajectory stress_scenarios =
  List.map (fun (scenario_name, stress_factors) ->
    let stressed_t = { t with 
      volatility = Tensor.mul t.volatility (Tensor.of_float stress_factors.volatility_multiplier);
      correlation = Tensor.mul t.correlation (Tensor.of_float stress_factors.correlation_multiplier);
      market_volume = Tensor.mul t.market_volume (Tensor.of_float stress_factors.volume_multiplier);
      bid_ask_spread = Tensor.mul t.bid_ask_spread (Tensor.of_float stress_factors.spread_multiplier);
      economic_factors = List.map (fun factor -> 
        { factor with value = factor.value *. stress_factors.economic_factor_multiplier }
      ) t.economic_factors;
    } in
    let stressed_cost = expected_cost stressed_t trajectory in
    let stressed_var = calculate_var stressed_t trajectory 0.95 in
    let stressed_cvar = calculate_cvar stressed_t trajectory 0.95 in
    (scenario_name, stressed_cost, stressed_var, stressed_cvar)
  ) stress_scenarios

let multi_period_optimization t =
  let num_assets = Tensor.size t.initial_portfolio 0 in
  let state = Tensor.cat [t.initial_portfolio; t.final_portfolio] ~dim:0 in
  
  let rec optimize remaining_periods state =
    if remaining_periods = 0 then
      []
    else
      let action = deep_rl_optimization t state in
      let new_state = Tensor.add (Tensor.slice state ~dim:0 ~start:0 ~end_:num_assets ~step:1) action in
      action :: optimize (remaining_periods - 1) (Tensor.cat [new_state; Tensor.slice state ~dim:0 ~start:num_assets ~end_:(2*num_assets) ~step:1] ~dim:0)
  in
  
  optimize t.num_periods state

let parallel_monte_carlo_simulation t trajectory num_simulations =
  let num_threads = 4 in
  let simulations_per_thread = num_simulations / num_threads in
  
  let simulate_batch () =
    let simulated_costs = Tensor.zeros [simulations_per_thread] in
    let simulated_vars = Tensor.zeros [simulations_per_thread] in
    let simulated_cvars = Tensor.zeros [simulations_per_thread] in
    
    for i = 0 to simulations_per_thread - 1 do
      let random_shock = Tensor.randn_like trajectory in
      let perturbed_trajectory = Tensor.add trajectory (Tensor.mul random_shock (Tensor.sqrt (cost_variance t trajectory))) in
      let perturbed_cost = expected_cost t perturbed_trajectory in
      let perturbed_var = calculate_var t perturbed_trajectory 0.95 in
      let perturbed_cvar = calculate_cvar t perturbed_trajectory 0.95 in
      Tensor.set simulated_costs [i] perturbed_cost;
      Tensor.set simulated_vars [i] perturbed_var;
      Tensor.set simulated_cvars [i] perturbed_cvar
    done;
    
    (simulated_costs, simulated_vars, simulated_cvars)
  in
  
  let batches = List.init num_threads (fun _ -> Lwt.apply simulate_batch ()) in
  let results = Lwt_main.run (Lwt.all batches) in
  
  let combined_costs = Tensor.cat (List.map (fun (c, _, _) -> c) results) ~dim:0 in
  let combined_vars = Tensor.cat (List.map (fun (_, v, _) -> v) results) ~dim:0 in
  let combined_cvars = Tensor.cat (List.map (fun (_, _, cv) -> cv) results) ~dim:0 in
  
  (combined_costs, combined_vars, combined_cvars)

let expected_cost t trajectory =
  let rates = trading_rate t trajectory in
  let adjusted_rates = liquidity_adjustment t rates in
  let network_adjusted_rates = network_impact_model t adjusted_rates trajectory in
  let impact = market_impact_model t network_adjusted_rates in
  let trans_cost = transaction_cost t network_adjusted_rates in
  Tensor.add (Tensor.sum (Tensor.mul network_adjusted_rates impact)) (Tensor.sum trans_cost)

let cost_variance t trajectory =
  let cov = calculate_covariance t in
  let tau = calculate_tau t in
  Tensor.sum (Tensor.mul (Tensor.of_float (tau ** 2.)) (Tensor.matmul (Tensor.matmul trajectory cov) (Tensor.transpose trajectory ~dim0:0 ~dim1:1)))

let calculate_var t trajectory confidence_level =
  let num_simulations = 10000 in
  let simulated_costs = Tensor.zeros [num_simulations] in
  for i = 0 to num_simulations - 1 do
    let random_shock = Tensor.randn_like trajectory in
    let perturbed_trajectory = Tensor.add trajectory (Tensor.mul random_shock (Tensor.sqrt (cost_variance t trajectory))) in
    let perturbed_cost = expected_cost t perturbed_trajectory in
    Tensor.set simulated_costs [i] perturbed_cost
  done;
  let sorted_costs = Tensor.sort simulated_costs ~descending:false in
  let var_index = int_of_float (float_of_int num_simulations *. (1. -. confidence_level)) in
  Tensor.get sorted_costs [var_index]

let calculate_cvar t trajectory confidence_level =
  let var = calculate_var t trajectory confidence_level in
  let num_simulations = 10000 in
  let simulated_costs = Tensor.zeros [num_simulations] in
  for i = 0 to num_simulations - 1 do
    let random_shock = Tensor.randn_like trajectory in
    let perturbed_trajectory = Tensor.add trajectory (Tensor.mul random_shock (Tensor.sqrt (cost_variance t trajectory))) in
    let perturbed_cost = expected_cost t perturbed_trajectory in
    Tensor.set simulated_costs [i] perturbed_cost
  done;
  let tail_costs = Tensor.masked_select simulated_costs (Tensor.gt simulated_costs var) in
  Tensor.mean tail_costs

let trading_rate t trajectory =
  Tensor.sub trajectory (Tensor.slice trajectory ~dim:0 ~start:1 ~end_:(Tensor.size trajectory 0) ~step:1)

let multi_asset_optimization t =
  let num_assets = Tensor.size t.initial_portfolio 0 in
  let num_periods = t.num_periods in
  
  let initial_trajectory = Tensor.zeros [num_periods + 1; num_assets] ~requires_grad:true in
  Tensor.copy_ (Tensor.get initial_trajectory [0]) t.initial_portfolio;
  Tensor.copy_ (Tensor.get initial_trajectory [-1]) t.final_portfolio;
  
  let optimizer = Optimizer.adam [initial_trajectory] ~lr:0.01 in
  
  let num_iterations = 1000 in
  for i = 1 to num_iterations do
    Optimizer.zero_grad optimizer;
    
    let trading_rates = Tensor.sub 
      (Tensor.slice initial_trajectory ~dim:0 ~start:0 ~end_:num_periods ~step:1)
      (Tensor.slice initial_trajectory ~dim:0 ~start:1 ~end_:(num_periods + 1) ~step:1)
    in
    
    let impact_cost = market_impact_model t trading_rates in
    let transaction_cost = Tensor.sum (transaction_cost t trading_rates) in
    
    let cov = calculate_covariance t in
    let risk = Tensor.sum (Tensor.matmul (Tensor.matmul initial_trajectory cov) (Tensor.transpose initial_trajectory ~dim0:0 ~dim1:1)) in
    
    let tau = calculate_tau t in
    let objective = Tensor.add (Tensor.add impact_cost transaction_cost) (Tensor.mul (Tensor.of_float (t.risk_aversion *. tau)) risk) in
    
    Tensor.backward objective;
    
    Optimizer.step optimizer;
    
    Tensor.copy_ (Tensor.get initial_trajectory [0]) t.initial_portfolio;
    Tensor.copy_ (Tensor.get initial_trajectory [-1]) t.final_portfolio;
    
    if i mod 100 = 0 then
      Printf.printf "Iteration %d, Objective: %f\n" i (Tensor.to_float0_exn objective)
  done;
  
  initial_trajectory

let calculate_total_cost t trajectory =
  let trading_rates = Tensor.sub 
    (Tensor.slice trajectory ~dim:0 ~start:0 ~end_:t.num_periods ~step:1)
    (Tensor.slice trajectory ~dim:0 ~start:1 ~end_:(t.num_periods + 1) ~step:1)
  in
  let impact_cost = market_impact_model t trading_rates in
  let transaction_cost = Tensor.sum (transaction_cost t trading_rates) in
  let cov = calculate_covariance t in
  let risk = Tensor.sum (Tensor.matmul (Tensor.matmul trajectory cov) (Tensor.transpose trajectory ~dim0:0 ~dim1:1)) in
  let tau = calculate_tau t in
  Tensor.add (Tensor.add impact_cost transaction_cost) (Tensor.mul (Tensor.of_float (t.risk_aversion *. tau)) risk)