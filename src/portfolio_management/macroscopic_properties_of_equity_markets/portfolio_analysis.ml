open Torch
open Analysis
open Ml_strategy

let calculate_diversity_weighted_portfolio p weights =
  let n = Array.length weights in
  let sum_p = Array.fold_left (fun acc w -> acc +. w ** p) 0. weights in
  Array.map (fun w -> (w ** p) /. sum_p) weights

let calculate_portfolio_return weights_t0 weights_t1 portfolio_weights =
    let n = Array.length weights_t0 in
    let return = ref 0. in
    for i = 0 to n - 1 do
      return := !return +. portfolio_weights.(i) *. (weights_t1.(i) /. weights_t0.(i) -. 1.)
    done;
    !return

let calculate_equal_weight_portfolio weights =
  Array.make (Array.length weights) (1. /. float_of_int (Array.length weights))

let calculate_market_cap_weighted_portfolio = Fun.id

let calculate_momentum_portfolio historical_weights lookback_period current_weights =
  let n = Array.length current_weights in
  let m = Array.length historical_weights in
  let momentum_scores = Array.make n 0. in
  for i = 0 to n - 1 do
    let start_index = max 0 (m - lookback_period) in
    let cumulative_return = ref 1. in
    for t = start_index to m - 1 do
      cumulative_return := !cumulative_return *. (historical_weights.(t).(i) /. historical_weights.(start_index).(i))
    done;
    momentum_scores.(i) <- !cumulative_return
  done;
  let total_score = Array.fold_left (+.) 0. momentum_scores in
  Array.map (fun score -> score /. total_score) momentum_scores

let calculate_sharpe_ratio returns risk_free_rate =
  let n = Array.length returns in
  let mean_return = Array.fold_left (+.) 0. returns /. float_of_int n in
  let variance = Array.fold_left (fun acc r -> 
    acc +. (r -. mean_return) ** 2.
  ) 0. returns /. float_of_int (n - 1) in
  let std_dev = sqrt variance in
  (mean_return -. risk_free_rate) /. std_dev

let calculate_maximum_drawdown values =
  let n = Array.length values in
  let rec loop i max_value max_drawdown =
    if i = n then max_drawdown
    else
      let new_max_value = max max_value values.(i) in
      let new_drawdown = (new_max_value -. values.(i)) /. new_max_value in
      let new_max_drawdown = max max_drawdown new_drawdown in
      loop (i + 1) new_max_value new_max_drawdown
  in
  loop 0 values.(0) 0.

let calculate_diversity_trend_portfolio historical_weights current_weights =
  let n = Array.length current_weights in
  let m = Array.length historical_weights in
  let lookback = min 20 m in
  let diversity_trend = 
    Array.init lookback (fun i -> 
      calculate_diversity_measure 0.5 historical_weights.(m - lookback + i)
    )
  in
  let trend = 
    if diversity_trend.(lookback-1) > diversity_trend.(0) then 1.
    else -1.
  in
  Array.mapi (fun i w -> 
    if trend > 0. then w ** 0.8
    else w ** 1.2
  ) current_weights

let calculate_volatility_responsive_portfolio historical_weights current_weights =
  let n = Array.length current_weights in
  let m = Array.length historical_weights in
  let lookback = min 20 m in
  let egr = calculate_excess_growth_rate_matrix (Array.sub historical_weights (m - lookback) lookback) in
  let avg_egr = Array.fold_left (+.) 0. egr /. float_of_int lookback in
  Array.mapi (fun i w -> 
    if avg_egr > 0.01 then w ** 0.8  (* Increase diversification when volatility is high *)
    else w ** 1.2  (* Concentrate more when volatility is low *)
  ) current_weights

let calculate_rank_momentum_portfolio historical_weights lookback_period current_weights =
  let n = Array.length current_weights in
  let m = Array.length historical_weights in
  let rank_changes = Array.make n 0 in
  for i = 0 to n - 1 do
    let start_rank = Market_analysis.get_rank historical_weights.(m - lookback_period - 1) i in
    let end_rank = Market_analysis.get_rank historical_weights.(m - 1) i in
    rank_changes.(i) <- start_rank - end_rank
  done;
  let total_rank_change = Array.fold_left (fun acc x -> acc + abs x) 0 rank_changes in
  Array.mapi (fun i w -> 
    let rank_score = float_of_int rank_changes.(i) /. float_of_int total_rank_change in
    w *. (1. +. rank_score)
  ) current_weights

let calculate_cross_sectional_momentum_portfolio returns lookback current_weights =
  let momentum = calculate_cross_sectional_momentum returns lookback in
  let last_momentum = Tensor.select momentum ~dim:0 ~index:(-1) in
  let scaled_momentum = Tensor.div last_momentum (Tensor.sum last_momentum) in
  Tensor.mul current_weights scaled_momentum

let calculate_factor_tilted_portfolio returns factors current_weights =
  let exposures = calculate_factor_exposures returns factors in
  let last_exposures = Tensor.select exposures ~dim:1 ~index:(-1) in
  let scaled_exposures = Tensor.div last_exposures (Tensor.sum last_exposures) in
  Tensor.mul current_weights scaled_exposures

let calculate_optimal_holding_portfolio returns current_weights =
  let max_period = 252 in (* Daily data, max 1 year holding period *)
  let optimal_period = int_of_float (estimate_optimal_holding_period returns max_period) in
  let n = Tensor.shape1_exn returns in
  let holding_returns = Tensor.zeros [n - optimal_period] in
  for t = optimal_period to n - 1 do
    let period_returns = Tensor.narrow returns ~dim:0 ~start:(t - optimal_period) ~length:optimal_period in
    let cumulative_returns = Tensor.prod period_returns ~dim:[0] in
    Tensor.index_put_ holding_returns [t - optimal_period] cumulative_returns
  done;
  let scaled_returns = Tensor.div holding_returns (Tensor.sum holding_returns) in
  Tensor.mul current_weights scaled_returns

let calculate_ml_portfolio historical_data current_weights =
  MLStrategy.calculate_ml_portfolio historical_data current_weights