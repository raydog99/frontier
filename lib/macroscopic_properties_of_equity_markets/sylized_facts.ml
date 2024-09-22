open Torch
open Market_analysis
open Analysis

let check_capital_distribution_stability market_caps =
  let n = Array.length market_caps in
  let stability_measure = ref 0. in
  for t = 1 to n - 1 do
    let dist_t = calculate_capital_distribution (calculate_market_weights market_caps.(t-1)) in
    let dist_t1 = calculate_capital_distribution (calculate_market_weights market_caps.(t)) in
    stability_measure := !stability_measure +. Tensor.(mean (abs (sub (of_float_array dist_t) (of_float_array dist_t1)))).float_value
  done;
  !stability_measure /. float_of_int (n - 1)

let check_market_diversity_behavior market_caps =
  let n = Array.length market_caps in
  let diversity_measures = Array.init n (fun t -> 
    calculate_diversity_measure 0.5 (calculate_market_weights market_caps.(t))
  ) in
  let diff = Array.init (n - 1) (fun t -> diversity_measures.(t+1) -. diversity_measures.(t)) in
  let mean_reversion = Tensor.(mean (mul (of_float_array diff) (of_float_array (Array.sub diff 1 (n-2))))).float_value
  mean_reversion

let check_excess_growth_rate_properties market_caps =
  let egr_matrix = calculate_excess_growth_rate_matrix market_caps in
  let egr_tensor = Tensor.of_float_array2 egr_matrix in
  let volatility_clustering = Tensor.(mean (abs (sub egr_tensor (mean egr_tensor ~dim:[0] ~keepdim:true)))).float_value in
  let correlation_with_market_vol = 
    let market_returns = Market_data.get_returns { market_caps; dates = [||] } in
    let market_vol = Tensor.(std (of_float_array2 market_returns) ~dim:[1]).float_value in
    Tensor.(mean (mul (of_float_array egr_matrix) (of_float_array market_vol))).float_value
  in
  (volatility_clustering, correlation_with_market_vol)

let check_rank_based_properties market_caps =
  let rank_volatility = calculate_rank_volatility market_caps in
  let rank_transition = calculate_rank_transition_probabilities market_caps in
  let rank_switching = calculate_rank_switching_intensity market_caps in
  let small_stock_volatility = Tensor.(mean (slice (of_float_array rank_volatility) ~dim:0 ~start:(Some (-100)) ~end_:None ~step:1)).float_value in
  let rank_stickiness = Tensor.(mean (diagonal (of_float_array2 rank_transition))).float_value in
  let switching_intensity_trend = 
    let n = Array.length rank_switching in
    Tensor.(sum (mul (of_float_array rank_switching) (of_float_array (Array.init n float_of_int)))).float_value /. float_of_int n
  in
  (small_stock_volatility, rank_stickiness, switching_intensity_trend)