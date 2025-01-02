open Torch

let calculate_diversity_measure p weights =
  let n = Array.length weights in
  let sum_p = Array.fold_left (fun acc w -> acc +. (w ** p)) 0. weights in
  sum_p ** (1. /. p)

let calculate_excess_growth_rate_matrix market_caps =
  let n = Array.length market_caps in
  let m = Array.length market_caps.(0) in
  Array.init (n - 1) (fun t ->
    let weights_t = Market_analysis.calculate_market_weights market_caps.(t) in
    let weights_t1 = Market_analysis.calculate_market_weights market_caps.(t + 1) in
    Market_analysis.calculate_excess_growth_rate weights_t weights_t1
  )

let estimate_rank_based_drift market_caps =
  let n = Array.length market_caps in
  let m = Array.length market_caps.(0) in
  let drift = Array.make m 0. in
  for i = 0 to m - 1 do
    let sum_drift = ref 0. in
    for t = 1 to n - 1 do
      let rank_t = Market_analysis.get_rank market_caps.(t-1) i in
      let rank_t1 = Market_analysis.get_rank market_caps.(t) i in
      sum_drift := !sum_drift +. (float_of_int (rank_t1 - rank_t))
    done;
    drift.(i) <- !sum_drift /. float_of_int (n - 1)
  done;
  drift

let estimate_rank_based_volatility market_caps =
  let n = Array.length market_caps in
  let m = Array.length market_caps.(0) in
  let volatility = Array.make m 0. in
  for i = 0 to m - 1 do
    let sum_squared_diff = ref 0. in
    for t = 1 to n - 1 do
      let rank_t = Market_analysis.get_rank market_caps.(t-1) i in
      let rank_t1 = Market_analysis.get_rank market_caps.(t) i in
      let diff = float_of_int (rank_t1 - rank_t) in
      sum_squared_diff := !sum_squared_diff +. (diff *. diff)
    done;
    volatility.(i) <- sqrt (!sum_squared_diff /. float_of_int (n - 1))
  done;
  volatility

let calculate_leakage market_caps k =
  let n = Array.length market_caps in
  let m = Array.length market_caps.(0) in
  let leakage = ref 0. in
  for t = 1 to n - 1 do
    let top_k_t = Market_analysis.get_top_k_stocks market_caps.(t-1) k in
    let top_k_t1 = Market_analysis.get_top_k_stocks market_caps.(t) k in
    let diff = List.filter (fun x -> not (List.mem x top_k_t1)) top_k_t in
    leakage := !leakage +. (float_of_int (List.length diff) /. float_of_int k)
  done;
  !leakage /. float_of_int (n - 1)

let calculate_intrinsic_volatility market_caps =
  let excess_growth_rates = calculate_excess_growth_rate_matrix market_caps in
  Array.fold_left (fun acc rate -> acc +. rate) 0. excess_growth_rates /. float_of_int (Array.length excess_growth_rates)

let calculate_cross_sectional_momentum returns lookback =
  let n, m = Tensor.shape2_exn returns in
  let momentum = Tensor.zeros [n - lookback; m] in
  for t = lookback to n - 1 do
    let period_returns = Tensor.narrow returns ~dim:0 ~start:(t - lookback) ~length:lookback in
    let cumulative_returns = Tensor.prod period_returns ~dim:[0] in
    Tensor.index_put_ momentum [t - lookback] cumulative_returns
  done;
  momentum

let calculate_factor_exposures returns factors =
  let n, m = Tensor.shape2_exn returns in
  let k = Tensor.shape1_exn factors in
  let x = Tensor.cat [Tensor.ones [n; 1]; factors] ~dim:1 in
  let xt = Tensor.transpose x ~dim0:0 ~dim1:1 in
  let xtx = Tensor.matmul xt x in
  let xtx_inv = Tensor.inverse xtx in
  let xty = Tensor.matmul xt returns in
  Tensor.matmul xtx_inv xty

let estimate_optimal_holding_period returns max_period =
  let n, m = Tensor.shape2_exn returns in
  let sharpe_ratios = Tensor.zeros [max_period] in
  for p = 1 to max_period do
    let holding_returns = Tensor.zeros [n - p; m] in
    for t = p to n - 1 do
      let period_returns = Tensor.narrow returns ~dim:0 ~start:(t - p) ~length:p in
      let cumulative_returns = Tensor.prod period_returns ~dim:[0] in
      Tensor.index_put_ holding_returns [t - p] cumulative_returns
    done;
    let mean_return = Tensor.mean holding_returns ~dim:[0] in
    let std_return = Tensor.std holding_returns ~dim:[0] in
    let sharpe = Tensor.div mean_return std_return in
    Tensor.index_put_ sharpe_ratios [p - 1] (Tensor.mean sharpe)
  done;
  Tensor.argmax sharpe_ratios ~dim:0 |> Tensor.float_value