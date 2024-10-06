open Torch

let run_analysis () =
  try
    (* Generate random price data for demonstration *)
    let prices = Tensor.rand [730; 5] in
    let symbols = [| "BTC/USD"; "ETH/USD"; "DAI/BTC"; "XRP/BTC"; "IOT/BTC" |] in
    let dataset = Dataset.create prices symbols in

    Printf.printf "Analyzing cryptocurrency time series...\n";
    let results = Analysis.analyze_lags_and_symbols dataset 20 5 in

    (* Aggregate results *)
    let aggregated_results = 
      List.fold_left (fun acc (lags, symbols, model, info, corr, mrr) ->
        let key = (lags, symbols, model) in
        let (sum_info, sum_corr, sum_mrr, count) =
          try Hashtbl.find acc key
          with Not_found -> (0., 0., 0., 0)
        in
        Hashtbl.replace acc key (sum_info +. info, sum_corr +. corr, sum_mrr +. mrr, count + 1);
        acc
      ) (Hashtbl.create 100) results
    in

    Printf.printf "\nAggregated Results:\n";
    Hashtbl.iter (fun (lags, symbols, model) (sum_info, sum_corr, sum_mrr, count) ->
      let avg_info = sum_info /. float_of_int count in
      let avg_corr = sum_corr /. float_of_int count in
      let avg_mrr = sum_mrr /. float_of_int count in
      Printf.printf "Lags: %d, Symbols: %d, Model: %s\n" lags symbols model;
      Printf.printf "  Avg. Information: %f bits\n" avg_info;
      Printf.printf "  Avg. Correlation: %f\n" avg_corr;
      Printf.printf "  Avg. MRR: %f%%\n\n" (avg_mrr *. 100.)
    ) aggregated_results;

    (* Find best configurations *)
    let best_info = ref (0., (0, 0, "")) in
    let best_corr = ref (0., (0, 0, "")) in
    let best_mrr = ref (0., (0, 0, "")) in

    Hashtbl.iter (fun (lags, symbols, model) (sum_info, sum_corr, sum_mrr, count) ->
      let avg_info = sum_info /. float_of_int count in
      let avg_corr = sum_corr /. float_of_int count in
      let avg_mrr = sum_mrr /. float_of_int count in
      
      if avg_info > fst !best_info then best_info := (avg_info, (lags, symbols, model));
      if avg_corr > fst !best_corr then best_corr := (avg_corr, (lags, symbols, model));
      if avg_mrr > fst !best_mrr then best_mrr := (avg_mrr, (lags, symbols, model))
    ) aggregated_results;

    Printf.printf "Best Configurations:\n";
    Printf.printf "Best Information: %f bits (Lags: %d, Symbols: %d, Model: %s)\n"
      (fst !best_info) (let (l, s, m) = snd !best_info in l) (let (l, s, m) = snd !best_info in s) (let (l, s, m) = snd !best_info in m);
    Printf.printf "Best Correlation: %f (Lags: %d, Symbols: %d, Model: %s)\n"
      (fst !best_corr) (let (l, s, m) = snd !best_corr in l) (let (l, s, m) = snd !best_corr in s) (let (l, s, m) = snd !best_corr in m);
    Printf.printf "Best MRR: %f%% (Lags: %d, Symbols: %d, Model: %s)\n"
      ((fst !best_mrr) *. 100.) (let (l, s, m) = snd !best_mrr in l) (let (l, s, m) = snd !best_mrr in s) (let (l, s, m) = snd !best_mrr in m);

    (* RMSE lower bound *)
    let best_info_value, (best_info_lags, best_info_symbols, _) = !best_info in
    let response = Dataset.create_response dataset best_info_lags in
    let sigma_x = Tensor.std response ~dim:[0] |> Tensor.to_float0_exn in
    let rmse_lower_bound = Voi.rmse_lower_bound best_info_value sigma_x in
    Printf.printf "\nRMSE Lower Bound: %f\n" rmse_lower_bound;

  with e ->
    Printf.eprintf "Error in analysis: %s\n" (Printexc.to_string e)