type performance = {
  return: float;
  risk: float;
  sharpe_ratio: float;
  max_drawdown: float;
}

let calculate_return portfolio_values =
  let initial_value = List.hd portfolio_values in
  let final_value = List.rev portfolio_values |> List.hd in
  (final_value /. initial_value) -. 1.0

let calculate_risk portfolio_returns =
  let mean_return = List.fold_left (+.) 0.0 portfolio_returns /. float (List.length portfolio_returns) in
  let squared_deviations = List.map (fun r -> (r -. mean_return) ** 2.0) portfolio_returns in
  sqrt (List.fold_left (+.) 0.0 squared_deviations /. float (List.length squared_deviations))

let calculate_sharpe_ratio portfolio_return risk risk_free_rate =
  (portfolio_return -. risk_free_rate) /. risk

let calculate_max_drawdown portfolio_values =
  let rec loop max_drawdown peak current_drawdown = function
    | [] -> max_drawdown
    | value :: rest ->
      let new_peak = max peak value in
      let new_drawdown = (new_peak -. value) /. new_peak in
      let new_max_drawdown = max max_drawdown new_drawdown in
      loop new_max_drawdown new_peak new_drawdown rest
  in
  loop 0.0 (List.hd portfolio_values) 0.0 (List.tl portfolio_values)

let evaluate_performance portfolio_values risk_free_rate =
  let portfolio_returns = List.map2 (fun v1 v2 -> (v2 -. v1) /. v1) portfolio_values (List.tl portfolio_values) in
  let portfolio_return = calculate_return portfolio_values in
  let risk = calculate_risk portfolio_returns in
  let sharpe_ratio = calculate_sharpe_ratio portfolio_return risk risk_free_rate in
  let max_drawdown = calculate_max_drawdown portfolio_values in
  { return = portfolio_return; risk; sharpe_ratio; max_drawdown }