open Torch
open Csv
open Error_handling

let load_data filename =
  try
    let csv_data = Csv.load filename in
    let data = List.map (fun row -> List.map float_of_string row) csv_data in
    let tensor = Tensor.of_float2 (Array.of_list (List.map Array.of_list data)) in
    info (Printf.sprintf "Data loaded from %s" filename);
    tensor
  with
  | Sys_error msg -> raise_error (Printf.sprintf "Failed to load data from %s: %s" filename msg)
  | _ -> raise_error (Printf.sprintf "Failed to process data from %s" filename)

let preprocess_data data =
  let mean = Tensor.mean data ~dim:[0] ~keepdim:true in
  let std = Tensor.std data ~dim:[0] ~keepdim:true ~unbiased:true in
  Tensor.((sub data mean) / std)

let evaluate_performance portfolio benchmark =
  let portfolio_return = Portfolio.calculate_portfolio_return portfolio in
  let benchmark_return = Tensor.mean benchmark in
  let excess_return = Tensor.(sub portfolio_return benchmark_return) in
  let portfolio_risk = Portfolio.calculate_portfolio_risk portfolio in
  let sharpe_ratio = Tensor.(div excess_return portfolio_risk) in
  Tensor.to_float0_exn sharpe_ratio

let split_data data train_ratio =
  let n = (Tensor.shape data).(0) in
  let train_size = int_of_float (float n *. train_ratio) in
  let train_data = Tensor.narrow data ~dim:0 ~start:0 ~length:train_size in
  let test_data = Tensor.narrow data ~dim:0 ~start:train_size ~length:(n - train_size) in
  (train_data, test_data)

let create_sequences data sequence_length =
  let n = (Tensor.shape data).(0) in
  let num_sequences = n - sequence_length + 1 in
  let sequences = Tensor.zeros [num_sequences; sequence_length; (Tensor.shape data).(1)] in
  for i = 0 to num_sequences - 1 do
    let sequence = Tensor.narrow data ~dim:0 ~start:i ~length:sequence_length in
    Tensor.copy_ (Tensor.select sequences ~dim:0 ~index:i) sequence
  done;
  sequences

let create_dataset data sequence_length =
  let sequences = create_sequences data sequence_length in
  let inputs = Tensor.narrow sequences ~dim:1 ~start:0 ~length:(sequence_length - 1) in
  let targets = Tensor.select sequences ~dim:1 ~index:(sequence_length - 1) in
  Dataset.create ~inputs ~targets