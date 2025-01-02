open Torch
open Error_handling

let construct_portfolio predictions aum threshold =
  let _, indices = Tensor.topk predictions ~k:(Tensor.shape predictions).(0) / 2 ~dim:0 in
  let selected_aum = Tensor.index_select aum ~dim:0 ~index:indices in
  let weights = Tensor.(div selected_aum (sum selected_aum)) in
  weights

let rebalance_portfolio current_portfolio predictions aum threshold =
  let new_portfolio = construct_portfolio predictions aum threshold in
  let diff = Tensor.(sub new_portfolio current_portfolio) in
  let buy = Tensor.relu diff in
  let sell = Tensor.relu (Tensor.neg diff) in
  (buy, sell)

let calculate_portfolio_return portfolio returns =
  Tensor.(sum (mul portfolio returns))

let calculate_portfolio_risk portfolio returns =
  let portfolio_return = calculate_portfolio_return portfolio returns in
  Tensor.(std portfolio_return ~dim:[0] ~unbiased:true)

let backtest_portfolio initial_portfolio predictions returns aum threshold =
  try
    let rec simulate current_portfolio acc_return day =
      if day >= (Tensor.shape returns).(0) then acc_return
      else
        let daily_return = Tensor.select returns ~dim:0 ~index:day in
        let portfolio_return = calculate_portfolio_return current_portfolio daily_return in
        let new_predictions = Tensor.select predictions ~dim:0 ~index:day in
        let new_portfolio = rebalance_portfolio current_portfolio new_predictions aum threshold |> fst in
        simulate new_portfolio (acc_return +. (Tensor.to_float0_exn portfolio_return)) (day + 1)
    in
    let final_return = simulate initial_portfolio 0. 0 in
    info (Printf.sprintf "Backtest completed. Final return: %.4f" final_return);
    final_return
  with
  | _ -> raise_error "Failed to backtest portfolio"

let calculate_portfolio_metrics portfolio returns benchmark =
  let portfolio_returns = calculate_portfolio_return portfolio returns in
  let excess_returns = Tensor.(sub portfolio_returns benchmark) in
  let sharpe_ratio = Polymodel.calculate_sharpe_ratio portfolio_returns (Tensor.zeros [1]) in
  let volatility = Tensor.(std portfolio_returns ~dim:[0] ~unbiased:true) in
  let max_drawdown = 
    let cumulative_returns = Tensor.(cumsum portfolio_returns ~dim:0) in
    let peak = Tensor.(cummax cumulative_returns ~dim:0) in
    Tensor.(min (div (sub peak cumulative_returns) peak))
  in
  (sharpe_ratio, volatility, max_drawdown)