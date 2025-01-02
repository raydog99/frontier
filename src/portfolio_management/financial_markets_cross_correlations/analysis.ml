open Types
open Torch
open Lwt.Infix

let detect_regime dccc_value threshold_bull threshold_bear =
  if dccc_value > threshold_bull then Bull
  else if dccc_value < threshold_bear then Bear
  else Neutral

let detect_regimes dccc_results threshold_bull threshold_bear =
  Array.map (fun result ->
    {
      timestamp = result.timestamp;
      regime = detect_regime result.dccc threshold_bull threshold_bear;
    }
  ) dccc_results

let calculate_eigenvalue_centrality adjacency_matrix =
  let n = Array.length adjacency_matrix in
  let v = Tensor.ones [n] in
  let rec power_iteration v iter max_iter epsilon =
    if iter >= max_iter then v
    else
      let new_v = Tensor.matmul adjacency_matrix v in
      let norm = Tensor.norm new_v in
      let normalized_v = Tensor.div new_v norm in
      if Tensor.norm (Tensor.sub normalized_v v) < epsilon then normalized_v
      else power_iteration normalized_v (iter + 1) max_iter epsilon
  in
  power_iteration v 0 100 1e-6

let identify_influential_markets mst market_indices =
  let n = Array.length mst in
  let centrality = calculate_eigenvalue_centrality (Tensor.of_float2 mst) in
  let centrality_list = Tensor.to_float1_exn centrality in
  Array.mapi (fun i c -> (market_indices.(i).name, c)) centrality_list
  |> Array.to_list
  |> List.sort (fun (_, c1) (_, c2) -> compare c2 c1)

let parallel_map f lst =
  let open Lwt.Infix in
  Lwt_list.map_p (fun x -> Lwt_preemptive.detach f x) lst

let calculate_hurst_exponent prices =
  let n = Array.length prices in
  let max_lag = min 100 (n / 4) in
  let prices_tensor = Tensor.of_float1 prices in
  let log_returns = Tensor.(log (slice prices_tensor [1,n]) - log (slice prices_tensor [0,n-1])) in
  
  let calculate_rs lag =
    let chunks = Tensor.split ~split_size:lag log_returns in
    let ranges = Tensor.map (fun chunk -> Tensor.max chunk - Tensor.min chunk) chunks in
    let stds = Tensor.map (fun chunk -> Tensor.std chunk ~dim:[0] ~unbiased:true ~keepdim:false) chunks in
    Tensor.mean (Tensor.div ranges stds)
  in
  
  let rs_values = parallel_map calculate_rs (List.init max_lag (fun i -> i + 1)) |> Lwt_main.run in
  
  let x = Tensor.log (Tensor.arange ~start:1. ~end_:(Float.of_int (max_lag + 1)) ~options:(Kind Float, Device CPU)) in
  let y = Tensor.log (Tensor.of_float1 (Array.of_list rs_values)) in
  
  let coeffs = Tensor.lstsq y x in
  Tensor.to_float0_exn (Tensor.get coeffs 0)

let analyze_market_efficiency market_indices =
  parallel_map (fun mi -> 
    let hurst = calculate_hurst_exponent mi.prices in
    { name = mi.name; hurst_exponent = hurst }
  ) (Array.to_list market_indices)
  |> Lwt_main.run
  |> Array.of_list

let calculate_volatility prices =
  let prices_tensor = Tensor.of_float1 prices in
  let returns = Tensor.(log (slice prices_tensor [1,Array.length prices]) - log (slice prices_tensor [0,Array.length prices - 1])) in
  Tensor.to_float0_exn (Tensor.std returns ~dim:[0] ~unbiased:true ~keepdim:false)

let detect_volatility_clusters prices window_size threshold =
  let prices_tensor = Tensor.of_float1 prices in
  let n = Array.length prices in
  let volatilities = Tensor.zeros [n - window_size + 1] in
  
  for i = 0 to n - window_size do
    let window = Tensor.narrow prices_tensor ~dim:0 ~start:i ~length:window_size in
    let volatility = calculate_volatility (Tensor.to_float1_exn window) in
    Tensor.set volatilities [i] volatility
  done;
  
  let mean_volatility = Tensor.mean volatilities in
  let std_volatility = Tensor.std volatilities ~dim:[0] ~unbiased:true ~keepdim:false in
  let normalized_volatilities = Tensor.((volatilities - mean_volatility) / std_volatility) in
  
  let rec find_clusters idx clusters current_cluster =
    if idx >= Tensor.shape normalized_volatilities |> List.hd then
      List.rev (if current_cluster <> None then current_cluster :: clusters else clusters)
    else
      let v = Tensor.get normalized_volatilities [idx] in
      if v > threshold then
        match current_cluster with
        | None -> find_clusters (idx + 1) clusters (Some (idx, idx, v))
        | Some (start, _, max_v) -> find_clusters (idx + 1) clusters (Some (start, idx, max max_v v))
      else
        match current_cluster with
        | None -> find_clusters (idx + 1) clusters None
        | Some (start, end_, max_v) ->
            find_clusters (idx + 1) ((start, end_, max_v) :: clusters) None
  in
  
  let clusters = find_clusters 0 [] None in
  Array.of_list (List.map (fun (start, end_, intensity) ->
    { start_timestamp = float_of_int start; end_timestamp = float_of_int end_; intensity }
  ) clusters)

let calculate_beta market_returns index_returns =
  let cov = Tensor.mean (Tensor.mul market_returns index_returns) - Tensor.mean market_returns *. Tensor.mean index_returns in
  let var = Tensor.var index_returns ~dim:[0] ~unbiased:true ~keepdim:false in
  Tensor.to_float0_exn (Tensor.div cov var)

let analyze_market_risk market_indices index =
  let index_returns = Tensor.(log (slice (of_float1 index.prices) [1,Array.length index.prices]) - 
                               log (slice (of_float1 index.prices) [0,Array.length index.prices - 1])) in
  Array.map (fun mi ->
    let market_returns = Tensor.(log (slice (of_float1 mi.prices) [1,Array.length mi.prices]) - 
                                  log (slice (of_float1 mi.prices) [0,Array.length mi.prices - 1])) in
    let beta = calculate_beta market_returns index_returns in
    (mi.name, beta)
  ) market_indices