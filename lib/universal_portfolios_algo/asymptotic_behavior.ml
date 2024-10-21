let asymptotic_growth_rate portfolio market_seq =
  let t = List.length market_seq - 1 in
  Portfolio.log_relative_value portfolio market_seq t /. float_of_int t

let relative_entropy_rate portfolio market_seq =
  let t = List.length market_seq - 1 in
  let cumulative_entropy = ref 0.0 in
  for i = 1 to t do
    let market = List.nth market_seq i in
    cumulative_entropy := !cumulative_entropy +. 
      FunctionallyGeneratedPortfolio.relative_entropy 
        (Portfolio.get_weights portfolio) 
        (Market.get_weights market)
  done;
  !cumulative_entropy /. float_of_int t

let asymptotic_variance portfolio market_seq =
  let t = List.length market_seq - 1 in
  let mean_growth_rate = asymptotic_growth_rate portfolio market_seq in
  let cumulative_squared_diff = ref 0.0 in
  for i = 1 to t do
    let growth_rate = Portfolio.log_relative_value portfolio market_seq i /. float_of_int i in
    cumulative_squared_diff := !cumulative_squared_diff +. (growth_rate -. mean_growth_rate) ** 2.0
  done;
  !cumulative_squared_diff /. float_of_int t

let law_of_large_numbers portfolio market_seq epsilon =
  let t = List.length market_seq - 1 in
  let mean_growth_rate = asymptotic_growth_rate portfolio market_seq in
  List.for_all (fun i ->
    let growth_rate = Portfolio.log_relative_value portfolio market_seq i /. float_of_int i in
    abs_float (growth_rate -. mean_growth_rate) < epsilon
  ) (List.init t (fun x -> x + 1))

let central_limit_theorem portfolio market_seq confidence_level =
  let t = List.length market_seq - 1 in
  let mean_growth_rate = asymptotic_growth_rate portfolio market_seq in
  let variance = asymptotic_variance portfolio market_seq in
  let z_score = 1.96 (* 95% confidence interval *) in
  let confidence_interval = (mean_growth_rate -. z_score *. sqrt variance, 
                             mean_growth_rate +. z_score *. sqrt variance) in
  List.for_all (fun i ->
    let growth_rate = Portfolio.log_relative_value portfolio market_seq i /. float_of_int i in
    growth_rate >= fst confidence_interval && growth_rate <= snd confidence_interval
  ) (List.init t (fun x -> x + 1))

let ergodic_theorem portfolio market_seq =
  let t = List.length market_seq - 1 in
  let time_average = asymptotic_growth_rate portfolio market_seq in
  let space_average = List.fold_left (fun acc market ->
    acc +. FunctionallyGeneratedPortfolio.relative_entropy (Portfolio.get_weights portfolio) (Market.get_weights market)
  ) 0.0 market_seq /. float_of_int t in
  abs_float (time_average -. space_average) < 1e-6

let lyapunov_exponent portfolio market_seq =
  let t = List.length market_seq - 1 in
  let cumulative_log_return = Portfolio.log_relative_value portfolio market_seq t in
  cumulative_log_return /. float_of_int t

let kolmogorov_sinai_entropy market_seq =
  let t = List.length market_seq - 1 in
  let n = Market.size (List.hd market_seq) in
  let entropy_rate = ref 0.0 in
  for i = 1 to t do
    let market = List.nth market_seq i in
    let prev_market = List.nth market_seq (i - 1) in
    entropy_rate := !entropy_rate -. 
      Array.fold_left2 (fun acc m pm ->
        acc +. (m /. pm) *. log (m /. pm)
      ) 0.0 (Market.get_weights market) (Market.get_weights prev_market)
  done;
  !entropy_rate /. float_of_int t

let pesin_formula portfolio market_seq =
  let lyapunov = lyapunov_exponent portfolio market_seq in
  let entropy = kolmogorov_sinai_entropy market_seq in
  abs_float (lyapunov -. entropy) < 1e-6