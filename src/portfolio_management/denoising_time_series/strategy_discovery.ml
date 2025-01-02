open Torch
open Config
open DataProcessing
open MutualLearning
open Logging

type t = {
  config: Config.t;
  mutual_learners: MutualLearning.t array;
  data: DataProcessing.t;
}

let create config =
  let data = DataProcessing.load_data config in
  let mutual_learners = Array.init config.num_contexts (fun _ -> 
    MutualLearning.create config.input_channels
  ) in
  { config; mutual_learners; data }

let train t =
  Array.iteri (fun i ml ->
    let input_data = DataProcessing.prepare_input t.data 0 i in
    MutualLearning.train ml input_data t.config.num_epochs t.config.learning_rate;
  ) t.mutual_learners

let calculate_trading_position t current_data context_index =
  let ml = t.mutual_learners.(context_index) in
  let codes = MutualLearning.extract_patterns ml current_data t.config.num_clusters in
  Tensor.rand [1]

let backtest t =
  let positions = Array.make t.config.num_contexts [||] in
  let returns = ref [] in
  
  for i = t.config.start_index to t.config.end_index do
    let context_positions = Array.init t.config.num_contexts (fun j ->
      let current_data = DataProcessing.prepare_input t.data i j in
      calculate_trading_position t current_data j
    ) in
    
    Array.iteri (fun j pos ->
      positions.(j) <- Array.append positions.(j) [|Tensor.float_value pos|]
    ) context_positions;
    
    let future_return = Tensor.float_value (Tensor.slice t.data.target ~dim:0 ~start:(i + t.data.horizon) ~end_:(i + t.data.horizon + 1)) in
    returns := future_return :: !returns
  done;
  
  (Array.map (fun pos -> Array.of_list (List.rev pos)) positions, Array.of_list (List.rev !returns))

let calculate_performance positions returns =
  let total_return = Array.fold_left (+.) 0. returns in
  let squared_returns = Array.map (fun r -> r *. r) returns in
  let variance = (Array.fold_left (+.) 0. squared_returns) /. float (Array.length returns) -. (total_return /. float (Array.length returns)) ** 2. in
  let sharpe_ratio = total_return /. sqrt variance in
  (total_return, sharpe_ratio)

let run t =
  try
    Logging.info "Starting strategy training";
    train t;
    Logging.info "Strategy training completed";
    
    Logging.info "Starting backtesting";
    let positions, returns = backtest t in
    Logging.info "Backtesting completed";
    
    let total_return, sharpe_ratio = calculate_performance positions.(0) returns in
    Logging.info (Printf.sprintf "Total Return: %.4f, Sharpe Ratio: %.4f" total_return sharpe_ratio);
    
    Ok (positions, returns)
  with
  | e -> Error (Printexc.to_string e)