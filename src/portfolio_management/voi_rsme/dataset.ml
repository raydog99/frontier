open Torch

type t = {
  prices: Tensor.t;
  log_returns: Tensor.t;
  symbols: string array;
}

let create prices symbols =
  if Tensor.size prices 1 <> Array.length symbols then
    failwith "Number of price columns must match number of symbols";
  let log_returns = Tensor.(log (slice prices [Some 1; None; None]) - log (slice prices [Some 0; -1; None])) in
  { prices; log_returns; symbols }

let split_train_test t train_size =
  let total_size = Tensor.size t.log_returns 0 in
  if train_size >= total_size then
    failwith "Train size must be less than total size";
  let train_data = Tensor.narrow t.log_returns ~dim:0 ~start:0 ~length:train_size in
  let test_data = Tensor.narrow t.log_returns ~dim:0 ~start:train_size ~length:(total_size - train_size) in
  train_data, test_data

let create_predictors t num_lags num_symbols =
  let total_size = Tensor.size t.log_returns 0 in
  let num_features = num_lags * num_symbols in
  if num_lags >= total_size then
    failwith "Number of lags must be less than total size";
  if num_symbols > Tensor.size t.log_returns 1 then
    failwith "Number of symbols must not exceed available symbols";
  let predictors = Tensor.zeros [total_size - num_lags; num_features] in
  for i = 0 to total_size - num_lags - 1 do
    for j = 0 to num_lags - 1 do
      for k = 0 to num_symbols - 1 do
        let idx = j * num_symbols + k in
        let value = Tensor.get t.log_returns [i + num_lags - j - 1; k] in
        Tensor.set predictors [i; idx] value
      done
    done
  done;
  predictors

let create_response t num_lags =
  if num_lags >= Tensor.size t.log_returns 0 then
    failwith "Number of lags must be less than total size";
  Tensor.narrow t.log_returns ~dim:0 ~start:num_lags ~length:(Tensor.size t.log_returns 0 - num_lags)

let create_rolling_windows t train_size test_size =
  let total_size = Tensor.size t.log_returns 0 in
  if train_size + test_size > total_size then
    failwith "Sum of train_size and test_size must not exceed total size";
  let num_windows = (total_size - train_size - test_size) / test_size + 1 in
  List.init num_windows (fun i ->
    let start = i * test_size in
    let train_data = Tensor.narrow t.log_returns ~dim:0 ~start ~length:train_size in
    let test_data = Tensor.narrow t.log_returns ~dim:0 ~start:(start + train_size) ~length:test_size in
    (train_data, test_data)
  )