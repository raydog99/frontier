open Torch

type time_series = Tensor.t

type cluster = {
  start_index: int;
  end_index: int;
  duration: int;
}

type probability_distribution = (int * float) list

type market_data = {
  asset_name: string;
  prices: time_series;
  returns: time_series;
  volatility: time_series;
}

let moving_average (series : time_series) (n : int) : time_series =
  if n <= 0 then invalid_arg "Moving average window size must be positive";
  let kernel = Tensor.ones [1; 1; n] ~device:(Tensor.device series) in
  let padded_series = Tensor.pad series ~pad:[0; n-1; 0; 0] ~mode:"reflect" in
  Tensor.conv1d padded_series kernel ~stride:1 ~padding:0
  |> Tensor.div (Tensor.of_float (float_of_int n))

let generate_clusters (series : time_series) (n : int) : cluster list =
  if Tensor.dim series <> 1 then invalid_arg "Input series must be 1-dimensional";
  let ma = moving_average series n in
  let diff = Tensor.sub series ma in
  let signs = Tensor.sign diff in
  let sign_changes = Tensor.abs (Tensor.sub signs (Tensor.roll signs ~shifts:[1] ~dims:[0])) in
  let indices = Tensor.nonzero sign_changes |> Tensor.squeeze ~dim:[1] in
  let indices_list = Tensor.to_float1 indices |> Array.to_list |> List.map int_of_float in
  
  let rec build_clusters acc start_idx = function
    | [] -> List.rev acc
    | [end_idx] -> 
        let cluster = {start_index = start_idx; end_index = end_idx; duration = end_idx - start_idx} in
        List.rev (cluster :: acc)
    | end_idx :: rest ->
        let cluster = {start_index = start_idx; end_index = end_idx; duration = end_idx - start_idx} in
        build_clusters (cluster :: acc) end_idx rest
  in
  
  build_clusters [] 0 indices_list

let calculate_cluster_probabilities (clusters : cluster list) : probability_distribution =
  let total_clusters = float_of_int (List.length clusters) in
  let duration_counts = 
    List.fold_left 
      (fun acc cluster -> 
        let count = try List.assoc cluster.duration acc with Not_found -> 0 in
        (cluster.duration, count + 1) :: List.remove_assoc cluster.duration acc)
      []
      clusters
  in
  List.map (fun (duration, count) -> (duration, float_of_int count /. total_clusters)) duration_counts

let kullback_leibler_divergence (p : probability_distribution) (q : probability_distribution) : float =
  let epsilon = 1e-10 in  (* Small value to avoid division by zero *)
  List.fold_left (fun acc (duration, p_prob) ->
    let q_prob = try List.assoc duration q with Not_found -> epsilon in
    if p_prob > epsilon then
      acc +. p_prob *. (log (p_prob /. q_prob))
    else
      acc
  ) 0.0 p

let kl_cluster_entropy (empirical_series : time_series) (model_series : time_series) (n_min : int) (n_max : int) : float =
  if n_min <= 0 || n_max < n_min then invalid_arg "Invalid n_min or n_max";
  let kl_divergence_n n =
    let empirical_clusters = generate_clusters empirical_series n in
    let model_clusters = generate_clusters model_series n in
    let p = calculate_cluster_probabilities empirical_clusters in
    let q = calculate_cluster_probabilities model_clusters in
    kullback_leibler_divergence p q
  in
  Array.init (n_max - n_min + 1) (fun i -> kl_divergence_n (i + n_min))
  |> Array.fold_left (+.) 0.0

let shannon_cluster_entropy (series : time_series) (n_min : int) (n_max : int) : float =
  if n_min <= 0 || n_max < n_min then invalid_arg "Invalid n_min or n_max";
  let shannon_entropy_n n =
    let clusters = generate_clusters series n in
    let probabilities = calculate_cluster_probabilities clusters in
    List.fold_left (fun acc (_, p) ->
      if p > 0.0 then acc -. p *. (log p) else acc
    ) 0.0 probabilities
  in
  Array.init (n_max - n_min + 1) (fun i -> shannon_entropy_n (i + n_min))
  |> Array.fold_left (+.) 0.0

let kl_cluster_weights (market_data_list : market_data list) (n_min : int) (n_max : int) : float list =
  if market_data_list = [] then invalid_arg "Empty market data list";
  let model_series = Tensor.randn (Tensor.shape (List.hd market_data_list).volatility) in
  let entropies = List.map (fun data -> kl_cluster_entropy data.volatility model_series n_min n_max) market_data_list in
  let total_entropy = List.fold_left (+.) 0.0 entropies in
  if total_entropy = 0.0 then invalid_arg "Total entropy is zero";
  List.map (fun entropy -> 1.0 /. entropy /. total_entropy) entropies

let realized_volatility (returns : time_series) (window : int) : time_series =
  if window <= 0 then invalid_arg "Volatility window must be positive";
  let squared_returns = Tensor.mul returns returns in
  let sum_squared_returns = moving_average squared_returns window in
  Tensor.sqrt (Tensor.mul sum_squared_returns (Tensor.of_float (float_of_int window)))

let process_market_data (file_path : string) (volatility_window : int) : market_data =
  let load_csv file_path =
    let ic = open_in file_path in
    let rec read_lines acc =
      try
        let line = input_line ic in
        let price = float_of_string (List.hd (String.split_on_char ',' line)) in
        read_lines (price :: acc)
      with End_of_file ->
        close_in ic;
        List.rev acc
    in
    read_lines []
  in
  let prices = load_csv file_path in
  let prices_tensor = Tensor.of_float1 (Array.of_list prices) in
  let returns = List.map2 (fun p1 p2 -> log (p2 /. p1)) prices (List.tl prices) in
  let returns_tensor = Tensor.of_float1 (Array.of_list returns) in
  let volatility = realized_volatility returns_tensor volatility_window in
  {
    asset_name = Filename.basename file_path;
    prices = prices_tensor;
    returns = returns_tensor;
    volatility = volatility;
  }

let optimal_portfolio (market_data_list : market_data list) (n_min : int) (n_max : int) (horizon : int) : float array =
  let weights = kl_cluster_weights market_data_list n_min n_max in
  let returns = List.map (fun data -> Tensor.narrow data.returns ~dim:0 ~start:(-horizon) ~length:horizon) market_data_list in
  let portfolio_return = List.fold_left2 (fun acc weight return ->
    Tensor.add acc (Tensor.mul_scalar return weight)
  ) (Tensor.zeros [horizon]) weights returns in
  Tensor.to_float1 portfolio_return

let load_multiple_assets (file_paths : string list) (volatility_window : int) : market_data list =
  List.map (fun path -> process_market_data path volatility_window) file_paths

let portfolio_statistics (returns : float array) : float * float * float =
  let mean_return = Array.fold_left (+.) 0. returns /. float_of_int (Array.length returns) in
  let squared_deviations = Array.map (fun r -> (r -. mean_return) ** 2.) returns in
  let volatility = sqrt (Array.fold_left (+.) 0. squared_deviations /. float_of_int (Array.length returns)) in
  let sharpe_ratio = mean_return /. volatility in
  (mean_return, volatility, sharpe_ratio)