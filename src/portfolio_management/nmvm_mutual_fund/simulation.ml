open Torch

type result = {
  portfolio: Portfolio.t;
  returns: float array;
  wealth: float array;
  performance: Performance.t;
}

type rebalancing_strategy =
  | NoRebalancing
  | PeriodicRebalancing of int
  | ThresholdRebalancing of float

let run model initial_portfolio initial_wealth risk_free_rate num_periods rebalancing_strategy optimization_strategy =
  if initial_wealth <= 0. then
    failwith "Initial wealth must be positive";
  if num_periods <= 0 then
    failwith "Number of periods must be positive";

  let returns = Array.make num_periods 0. in
  let wealth = Array.make (num_periods + 1) initial_wealth in
  let mutable portfolio = initial_portfolio in

  for t = 0 to num_periods - 1 do
    let r = Nmvm.sample model in
    returns.(t) <- Tensor.(dot portfolio r |> to_float0);
    wealth.(t + 1) <- wealth.(t) *. (1. +. returns.(t));

    (match rebalancing_strategy with
     | NoRebalancing -> ()
     | PeriodicRebalancing period when (t + 1) mod period = 0 ->
         portfolio <- Optimizer.optimize model optimization_strategy None initial_wealth risk_free_rate
     | ThresholdRebalancing threshold ->
         let current_weights = Tensor.(div_scalar portfolio (sum portfolio |> to_float0)) in
         if Tensor.(sum (abs (sub current_weights portfolio)) |> to_float0) > threshold then
           portfolio <- Optimizer.optimize model optimization_strategy None initial_wealth risk_free_rate
     | _ -> ())
  done;

  let performance = Performance.calculate model portfolio initial_wealth risk_free_rate num_periods None in
  { portfolio; returns; wealth; performance }

let compare_strategies model strategies initial_wealth risk_free_rate num_periods num_simulations rebalancing_strategy =
  if num_simulations <= 0 then
    failwith "Number of simulations must be positive";

  List.map (fun (name, strategy) ->
    let results = List.init num_simulations (fun _ ->
      let initial_portfolio = Optimizer.optimize model strategy None initial_wealth risk_free_rate in
      run model initial_portfolio initial_wealth risk_free_rate num_periods rebalancing_strategy strategy
    ) in
    let avg_performance = Performance.{
      cumulative_return = List.fold_left (fun acc r -> acc +. r.performance.cumulative_return) 0. results /. float_of_int num_simulations;
      annualized_return = List.fold_left (fun acc r -> acc +. r.performance.annualized_return) 0. results /. float_of_int num_simulations;
      annualized_volatility = List.fold_left (fun acc r -> acc +. r.performance.annualized_volatility) 0. results /. float_of_int num_simulations;
      sharpe_ratio = List.fold_left (fun acc r -> acc +. r.performance.sharpe_ratio) 0. results /. float_of_int num_simulations;
      sortino_ratio = List.fold_left (fun acc r -> acc +. r.performance.sortino_ratio) 0. results /. float_of_int num_simulations;
      maximum_drawdown = List.fold_left (fun acc r -> acc +. r.performance.maximum_drawdown) 0. results /. float_of_int num_simulations;
      value_at_risk = List.fold_left (fun acc r -> acc +. r.performance.value_at_risk) 0. results /. float_of_int num_simulations;
      conditional_value_at_risk = List.fold_left (fun acc r -> acc +. r.performance.conditional_value_at_risk) 0. results /. float_of_int num_simulations;
      calmar_ratio = List.fold_left (fun acc r -> acc +. r.performance.calmar_ratio) 0. results /. float_of_int num_simulations;
      omega_ratio = List.fold_left (fun acc r -> acc +. r.performance.omega_ratio) 0. results /. float_of_int num_simulations;
      information_ratio = List.fold_left (fun acc r -> acc +. r.performance.information_ratio) 0. results /. float_of_int num_simulations;
      treynor_ratio = List.fold_left (fun acc r -> acc +. r.performance.treynor_ratio) 0. results /. float_of_int num_simulations;
    } in
    (name, { 
      portfolio = List.hd results |> fun r -> r.portfolio;
      returns = Array.make num_periods 0.;
      wealth = Array.make (num_periods + 1) initial_wealth;
      performance = avg_performance 
    })
  ) strategies