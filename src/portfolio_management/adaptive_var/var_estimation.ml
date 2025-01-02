open Torch

type method_t =
  | HistoricalSimulation
  | VarianceCovariance
  | MonteCarlo

let historical_simulation losses confidence_level =
  let sorted_losses = Tensor.sort losses ~dim:0 ~descending:true in
  let index = int_of_float (float_of_int (Tensor.shape losses |> List.hd) *. (1. -. confidence_level)) in
  Tensor.select sorted_losses ~dim:0 ~index

let variance_covariance losses confidence_level =
  let mean = Tensor.mean losses in
  let std = Tensor.std losses ~dim:[0] in
  let z = match confidence_level with
    | 0.90 -> 1.282
    | 0.95 -> 1.645
    | 0.99 -> 2.326
    | _ -> invalid_arg "Unsupported confidence level" in
  Tensor.(mean + (f z * std))

let monte_carlo losses confidence_level num_simulations =
  let mean = Tensor.mean losses in
  let std = Tensor.std losses ~dim:[0] in
  let simulated_losses = Tensor.(mean + (std * randn [num_simulations])) in
  historical_simulation simulated_losses confidence_level

let estimate_var method_type losses confidence_level =
  match method_type with
  | HistoricalSimulation -> historical_simulation losses confidence_level
  | VarianceCovariance -> variance_covariance losses confidence_level
  | MonteCarlo -> monte_carlo losses confidence_level 10000