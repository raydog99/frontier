open Torch
open Csv

exception Invalid_data_shape of string

let normalize_prices prices =
  let open Tensor in
  let n = size prices |> List.hd in
  if n < 2 then raise (Invalid_data_shape "Price data must have at least 2 time steps");
  let base_price = slice prices ~dim:0 ~start:1 ~end_:2 in
  let normalized = sub prices base_price |> div base_price in
  cat [zeros [1; 4]; normalized] ~dim:0

let preprocess_data data =
  let open Tensor in
  match size data with
  | [m; _; n] when n >= 4 ->
    let processed = Tensor.map normalize_prices data in
    let noise = randn [m; 4; n] ~dtype:(kind processed) in
    add processed (mul noise (float_scalar 0.001))
  | _ -> raise (Invalid_data_shape "Data must be of shape [m; 4; n] where n >= 4")

let load_and_preprocess_csv filename =
  let data = Csv.load ~separator:',' filename in
  let headers = List.hd data in
  let data_without_headers = List.tl data in
  let tensor_data = Tensor.of_float2 (List.map (List.map float_of_string) data_without_headers) in
  let reshaped_data = Tensor.reshape tensor_data [-1; 4; List.length headers / 4] in
  preprocess_data reshaped_data
