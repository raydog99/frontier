open Torch

let standardize tensor =
  let mean = Tensor.mean tensor ~dim:[0] ~keepdim:true in
  let std = Tensor.std tensor ~dim:[0] ~keepdim:true in
  Tensor.((tensor - mean) / std)

let dwt tensor threshold =
  let rec dwt_recursive tensor level =
    if level = 0 then tensor
    else
      let low, high = Tensor.split tensor ~dim:0 ~sizes:[Tensor.shape tensor |> List.hd |> (fun x -> x / 2); -1] in
      let filtered_high = Tensor.where_ (Tensor.(abs high) > threshold) high (Tensor.zeros_like high) in
      dwt_recursive (Tensor.cat [low; filtered_high] ~dim:0) (level - 1)
  in
  dwt_recursive tensor 3  (* Perform 3 levels of DWT *)

let preprocess_price_volume price volume =
  let price = standardize price |> dwt 0.7 in
  let volume = standardize volume |> dwt 0.7 in
  { Cet.price; volume }

let preprocess_earnings earnings =
  standardize earnings

let create_sequences tensor sequence_length =
  let num_sequences = Tensor.shape tensor |> List.hd |> (fun x -> x - sequence_length + 1) in
  Tensor.slice tensor ~dim:0 ~start:0 ~end_:num_sequences ~step:1
  |> Tensor.unfold ~dim:0 ~size:sequence_length ~step:1