open Torch
open Types

let generate_random_assets n_assets n_timepoints =
  List.init n_assets (fun i ->
    let symbol = Printf.sprintf "ASSET_%d" i in
    let prices = Tensor.rand [n_timepoints] in
    { symbol; prices }
  )

let tensor_to_list tensor =
  let n = Tensor.size tensor 0 in
  List.init n (fun i -> Tensor.get tensor [i] |> Tensor.item)

let print_tensor_statistics name tensor =
  let stats = Tensor.{
    mean = mean tensor;
    std = std tensor;
    min = min tensor;
    max = max tensor;
  } in
  Printf.printf "%s statistics:\n" name;
  Printf.printf "  Mean: %f\n" (Tensor.item stats.mean);
  Printf.printf "  Std:  %f\n" (Tensor.item stats.std);
  Printf.printf "  Min:  %f\n" (Tensor.item stats.min);
  Printf.printf "  Max:  %f\n" (Tensor.item stats.max)