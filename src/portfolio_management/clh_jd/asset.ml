open Torch

type t = {
  initial_price : float;
  drift : float;
  volatility : float;
  jump_intensity : float;
  jump_size : float;
}

let create ~initial_price ~drift ~volatility ~jump_intensity ~jump_size =
  { initial_price; drift; volatility; jump_intensity; jump_size }

let simulate_price asset ~time_steps ~dt =
  let open Tensor in
  let n = time_steps + 1 in
  let brownian_increments = randn [n] ~kind:(T Float) in
  let jump_occurrences = rand [n] ~kind:(T Float) in
  
  let drift_term = float asset.drift *. dt |> full [n] in
  let diffusion_term = float asset.volatility *. sqrt dt |> full [n] |> mul brownian_increments in
  let jump_term = 
    jump_occurrences 
    |> le (float asset.jump_intensity *. dt |> full [n])
    |> float_value
    |> mul (log (1. +. asset.jump_size) |> full [n])
  in
  
  let log_returns = add (add drift_term diffusion_term) jump_term in
  let cumulative_returns = cumsum log_returns ~dim:0 ~dtype:(T Float) in
  
  full [n] (float asset.initial_price)
  |> mul (exp cumulative_returns)