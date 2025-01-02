open Torch
open Optimization
open Parameter_shift
open Trading_strategy

type t = {
  initial_holdings: float;
  liquidation_time: float;
  num_intervals: int;
  volatility: float;
  permanent_impact: float;
  temporary_impact: float;
  drift: float;
  risk_aversion_range: float * float;
  parameter_shift: Parameter_shift.t option;
}

let create initial_holdings liquidation_time num_intervals volatility permanent_impact temporary_impact drift max_risk_aversion ?parameter_shift () =
  { initial_holdings; liquidation_time; num_intervals; volatility; permanent_impact; temporary_impact; drift; risk_aversion_range = (0., max_risk_aversion); parameter_shift }

let calculate_frontier t num_points =
  let { initial_holdings; liquidation_time; num_intervals; volatility; permanent_impact; temporary_impact; drift; risk_aversion_range; _ } = t in
  let (min_risk_aversion, max_risk_aversion) = risk_aversion_range in
  
  let risk_aversion_values = Tensor.linspace ~start:min_risk_aversion ~end_:max_risk_aversion ~steps:num_points in
  
  let optimize_for_risk_aversion risk_aversion =
    let optimization = Optimization.create initial_holdings liquidation_time num_intervals risk_aversion volatility permanent_impact temporary_impact drift in
    let strategy = Trading_strategy.create initial_holdings liquidation_time num_intervals in
    Optimization.optimize optimization (Trading_strategy.get_trading_trajectory strategy)
  in

  let optimal_trajectories = Tensor.map risk_aversion_values ~f:optimize_for_risk_aversion in

  let calculate_expected_cost trajectory =
    let optimization = Optimization.create initial_holdings liquidation_time num_intervals 0. volatility permanent_impact temporary_impact drift in
    Optimization.objective_function optimization trajectory
  in

  let calculate_variance trajectory =
    let interval_length = liquidation_time /. float_of_int num_intervals in
    Tensor.(sum (pow trajectory (f 2.)) * (f (volatility *. volatility *. interval_length)))
  in

  let expected_costs = Tensor.map optimal_trajectories ~f:calculate_expected_cost in
  let variances = Tensor.map optimal_trajectories ~f:calculate_variance in

  (expected_costs, variances)

let optimize_utility t utility =
  let { initial_holdings; liquidation_time; num_intervals; volatility; permanent_impact; temporary_impact; drift; _ } = t in
  let risk_aversion = Utility.risk_aversion utility in
  let optimization = Optimization.create initial_holdings liquidation_time num_intervals risk_aversion volatility permanent_impact temporary_impact drift in
  let strategy = Trading_strategy.create initial_holdings liquidation_time num_intervals in
  Optimization.optimize optimization (Trading_strategy.get_trading_trajectory strategy)

let calculate_var t confidence_level =
  let (expected_costs, variances) = calculate_frontier t 100 in
  let var_values = Tensor.map2 expected_costs variances ~f:(fun ec v -> 
    ValueAtRisk.calculate ec v confidence_level
  ) in
  let min_var_index = Tensor.argmin var_values ~dim:0 ~keepdim:false in
  let min_var = Tensor.get var_values (Tensor.to_int0_exn min_var_index) in
  let optimal_trajectory = optimize_utility t (Utility.create (Tensor.get expected_costs (Tensor.to_int0_exn min_var_index))) in
  (optimal_trajectory, min_var)

let optimize_time_dependent t strategy =
  let { initial_holdings; liquidation_time; num_intervals; volatility; permanent_impact; temporary_impact; drift; _ } = t in
  let optimization = Optimization.create initial_holdings liquidation_time num_intervals 0. volatility permanent_impact temporary_impact drift in
  Optimization.optimize optimization (TimeDependentStrategy.get_trading_trajectory strategy)

let analyze_strategy t trajectory =
  let { initial_holdings; liquidation_time; num_intervals; volatility; permanent_impact; temporary_impact; drift; parameter_shift; _ } = t in
  
  let interval_length = liquidation_time /. float_of_int num_intervals in
  let price_dynamics = PriceDynamics.create initial_holdings volatility drift ?parameter_shift () in
  let prices = PriceDynamics.simulate_price price_dynamics 0. liquidation_time num_intervals in
  
  let implementation_shortfall = Analysis.calculate_implementation_shortfall trajectory prices initial_holdings in
  let vwap = Analysis.calculate_vwap trajectory prices in
  let participation_rate = Analysis.calculate_participation_rate trajectory (initial_holdings /. liquidation_time) in
  
  Printf.printf "Strategy Analysis:\n";
  Printf.printf "Implementation Shortfall: %f\n" implementation_shortfall;
  Printf.printf "VWAP: %f\n" vwap;
  Printf.printf "Average Participation Rate: %f\n" (Array.fold_left (+.) 0. participation_rate /. float_of_int (Array.length participation_rate))

let optimize_almgren_chriss t =
  let { initial_holdings; liquidation_time; num_intervals; volatility; permanent_impact; temporary_impact; drift; _ } = t in
  let risk_aversion = snd t.risk_aversion_range in
  let optimization = Optimization.create initial_holdings liquidation_time num_intervals risk_aversion volatility permanent_impact temporary_impact drift in
  let initial_trajectory = Trading_strategy.get_trading_trajectory (Trading_strategy.create initial_holdings liquidation_time num_intervals) in
  Optimization.optimize_almgren_chriss optimization initial_trajectory

let optimize_dynamic t price_dynamics =
  let { initial_holdings; liquidation_time; num_intervals; volatility; permanent_impact; temporary_impact; drift; _ } = t in
  let risk_aversion = snd t.risk_aversion_range in
  let optimization = Optimization.create initial_holdings liquidation_time num_intervals risk_aversion volatility permanent_impact temporary_impact drift in
  let initial_trajectory = Trading_strategy.get_trading_trajectory (Trading_strategy.create initial_holdings liquidation_time num_intervals) in
  Optimization.optimize_dynamic optimization initial_trajectory price_dynamics

let optimize_constrained t constraint_fn =
  let { initial_holdings; liquidation_time; num_intervals; volatility; permanent_impact; temporary_impact; drift; _ } = t in
  let risk_aversion = snd t.risk_aversion_range in
  let optimization = Optimization.create initial_holdings liquidation_time num_intervals risk_aversion volatility permanent_impact temporary_impact drift in
  let initial_trajectory = Trading_strategy.get_trading_trajectory (Trading_strategy.create initial_holdings liquidation_time num_intervals) in
  Optimization.optimize_constrained optimization initial_trajectory constraint_fn