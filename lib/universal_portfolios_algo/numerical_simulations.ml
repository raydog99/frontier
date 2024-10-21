let simulate_market initial_weights mu sigma gamma dt n_steps n_simulations =
  List.init n_simulations (fun _ ->
    Market.rank_based_diffusion initial_weights mu sigma gamma dt n_steps
  )

let verify_asymptotic_universality portfolios market_seqs =
  let n_simulations = List.length market_seqs in
  let t = List.length (List.hd market_seqs) - 1 in
  let universal_portfolio = WealthDistribution.create portfolios in
  let best_constant_rebalanced = ref (List.hd portfolios) in
  List.iter (fun market_seq ->
    let updated_dist = WealthDistribution.update universal_portfolio market_seq t in
    best_constant_rebalanced := List.fold_left (fun best p ->
      if Portfolio.relative_value (FunctionallyGeneratedPortfolio.generate_weights p (List.hd market_seq)) market_seq t > 
         Portfolio.relative_value (FunctionallyGeneratedPortfolio.generate_weights best (List.hd market_seq)) market_seq t
      then p else best
    ) !best_constant_rebalanced portfolios
  ) market_seqs;
  let universal_performance = List.map (fun market_seq ->
    let updated_dist = WealthDistribution.update universal_portfolio market_seq t in
    Portfolio.relative_value (WealthDistribution.covers_universal_portfolio updated_dist market_seq t) market_seq t
  ) market_seqs in
  let best_crp_performance = List.map (fun market_seq ->
    Portfolio.relative_value (FunctionallyGeneratedPortfolio.generate_weights !best_constant_rebalanced (List.hd market_seq)) market_seq t
  ) market_seqs in
  let ratio = List.map2 (fun u b -> u /. b) universal_performance best_crp_performance in
  let avg_ratio = List.fold_left (+.) 0.0 ratio /. float_of_int n_simulations in
  Printf.printf "Average performance ratio: %.4f\n" avg_ratio;
  avg_ratio > 0.99

let verify_relative_arbitrage market_seqs epsilon =
  let n_simulations = List.length market_seqs in
  let t = List.length (List.hd market_seqs) - 1 in
  let n = Market.size (List.hd (List.hd market_seqs)) in
  let initial_weights = Array.make n (1.0 /. float_of_int n) in
  let arbitrage_portfolio = FunctionallyGeneratedPortfolio.diversity_weighted 0.5 in
  let arbitrage_performance = List.map (fun market_seq ->
    Portfolio.relative_value (FunctionallyGeneratedPortfolio.generate_weights arbitrage_portfolio (List.hd market_seq)) market_seq t
  ) market_seqs in
  let market_performance = List.map (fun market_seq ->
    Portfolio.relative_value (Portfolio.create n initial_weights) market_seq t
  ) market_seqs in
  let outperformance = List.map2 (fun a m -> a > m *. (1.0 +. epsilon)) arbitrage_performance market_performance in
  let success_rate = List.fold_left (fun acc b -> if b then acc +. 1.0 else acc) 0.0 outperformance /. float_of_int n_simulations in
  Printf.printf "Relative arbitrage success rate: %.4f\n" success_rate;
  success_rate > 0.95