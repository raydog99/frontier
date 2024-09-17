open Torch

let device = if Cuda.is_available () then Device.Cuda else Device.Cpu

let to_device tensor = Tensor.to_device tensor device

let read_csv filename =
  try
    let ic = open_in filename in
    let rec read_lines acc =
      try
        let line = input_line ic in
        let values = String.split_on_char ',' line in
        read_lines (values :: acc)
      with End_of_file ->
        close_in ic;
        List.rev acc
    in
    Ok (read_lines [])
  with
  | Sys_error msg -> Error (Printf.sprintf "Failed to open file: %s" msg)
  | e -> Error (Printf.sprintf "Unexpected error while reading CSV: %s" (Printexc.to_string e))

let to_tensor list =
  try
    let array = Array.of_list list in
    let float_array = Array.map (Array.map float_of_string) array in
    Ok (Tensor.of_float2 float_array |> to_device)
  with
  | Failure msg -> Error (Printf.sprintf "Failed to convert to tensor: %s" msg)
  | e -> Error (Printf.sprintf "Unexpected error while converting to tensor: %s" (Printexc.to_string e))

let split_data data train_ratio =
  if train_ratio <= 0.0 || train_ratio >= 1.0 then
    Error "Invalid train ratio. Must be between 0 and 1."
  else
    let len = List.length data in
    let train_len = int_of_float (float_of_int len *. train_ratio) in
    if train_len = 0 || train_len = len then
      Error "Train/test split resulted in empty set. Adjust ratio or increase data size."
    else
      let train, test = List.split_at train_len data in
      Ok (train, test)

let evaluate_sharpe_ratio portfolio_weights returns covariance =
  try
    let portfolio_return = Tensor.(sum (portfolio_weights * returns)) in
    let portfolio_risk = Tensor.(sqrt (matmul (matmul portfolio_weights (transpose covariance ~dim0:0 ~dim1:1)) portfolio_weights)) in
    Ok Tensor.(portfolio_return / portfolio_risk)
  with
  | e -> Error (Printf.sprintf "Error calculating Sharpe ratio: %s" (Printexc.to_string e))

let calculate_covariance returns =
  try
    let centered = Tensor.(returns - mean returns ~dim:[0] ~keepdim:true) in
    let cov = Tensor.(matmul (transpose centered ~dim0:0 ~dim1:1) centered) in
    Ok Tensor.(div cov (float (Tensor.shape returns |> List.hd - 1)))
  with
  | e -> Error (Printf.sprintf "Error calculating covariance: %s" (Printexc.to_string e))