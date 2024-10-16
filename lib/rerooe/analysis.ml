let calculate_var results confidence_level =
  let sorted_returns = List.sort (fun (a, _, _) (b, _, _) -> compare a b) (List.flatten results) in
  let n = List.length sorted_returns in
  let index = int_of_float (float_of_int n *. (1. -. confidence_level)) in
  let (var, _, _) = List.nth sorted_returns index in
  -.var

let calculate_expected_shortfall results confidence_level =
  let sorted_returns = List.sort (fun (a, _, _) (b, _, _) -> compare a b) (List.flatten results) in
  let n = List.length sorted_returns in
  let cutoff_index = int_of_float (float_of_int n *. (1. -. confidence_level)) in
  let sum_tail = List.fold_left (fun acc (pnl, _, _) -> acc +. pnl) 0. (List.filteri (fun i _ -> i < cutoff_index) sorted_returns) in
  -.(sum_tail /. float_of_int cutoff_index)

let calculate_max_drawdown results =
  let max_drawdown = ref 0. in
  let peak = ref neg_infinity in
  List.iter (fun path ->
    let path_peak = ref neg_infinity in
    let path_max_drawdown = ref 0. in
    List.iter (fun (pnl, _, _) ->
      if pnl > !path_peak then path_peak := pnl
      else
        let drawdown = (!path_peak -. pnl) /. !path_peak in
        if drawdown > !path_max_drawdown then path_max_drawdown := drawdown
    ) path;
    if !path_max_drawdown > !max_drawdown then max_drawdown := !path_max_drawdown
  ) results;
  !max_drawdown

let calculate_sharpe_ratio results =
  let returns = List.map (fun (pnl, _, _) -> pnl) (List.flatten results) in
  let mean_return = Statistics.mean returns in
  let std_dev = Statistics.standard_deviation returns in
  if std_dev = 0. then 0. else mean_return /. std_dev

let generate_report results strategy_name =
  let avg, std_dev = Simulation.calculate_statistics results in
  let sharpe_ratio = calculate_sharpe_ratio results in
  let max_drawdown = calculate_max_drawdown results in
  let var_95 = calculate_var results 0.95 in
  let es_95 = calculate_expected_shortfall results 0.95 in
  
  Printf.printf "Advanced Analysis Report for %s Strategy\n" strategy_name;
  Printf.printf "---------------------------------------------\n";
  Printf.printf "Sharpe Ratio: %.4f\n" sharpe_ratio;
  Printf.printf "Max Drawdown: %.2f%%\n" (max_drawdown *. 100.);
  Printf.printf "Value at Risk (95%%): %.2f\n" var_95;
  Printf.printf "Expected Shortfall (95%%): %.2f\n" es_95;
  Printf.printf "Final PnL: %.2f ± %.2f\n" (fst (List.hd (List.rev avg))) (fst (List.hd (List.rev std_dev)));
  Printf.printf "---------------------------------------------\n"