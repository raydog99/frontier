open Torch
open Types
open Garch_core

type mgarch_model =
  | DCC
  | BEKK

let estimate_dcc_garch returns n_assets max_iter learning_rate =
  let univariate_models = Array.init n_assets (fun i ->
    let asset_returns = Tensor.select returns ~dim:1 ~index:i in
    estimate_garch_parameters_generic GARCH asset_returns ~max_iter ~learning_rate
  ) in
  
  let standardized_returns = Tensor.zeros_like returns in
  for i = 0 to n_assets - 1 do
    let asset_returns = Tensor.select returns ~dim:1 ~index:i in
    let omega, alpha, _, beta = univariate_models.(i) in
    let vol = forecast_volatility GARCH (omega, alpha, 0.0, beta) asset_returns (Tensor.shape asset_returns |> List.hd) in
    Tensor.set standardized_returns ~dim:1 ~index:i Tensor.(asset_returns / sqrt vol);
  done;

  let a = Tensor.rand [] ~dtype:Tensor.Float in
  let b = Tensor.rand [] ~dtype:Tensor.Float in
  
  for _ = 1 to max_iter do
    let q_bar = Tensor.(matmul (transpose standardized_returns) standardized_returns) in
    let q = Tensor.zeros_like q_bar in
    Tensor.set q [0] q_bar;
    
    for t = 1 to (Tensor.shape standardized_returns |> List.hd) - 1 do
      let prev_q = Tensor.get q [t-1] in
      let prev_std = Tensor.get standardized_returns [t-1] in
      let new_q = Tensor.((Scalar.f (1.0 -. Tensor.to_float0_exn a -. Tensor.to_float0_exn b) * q_bar) +
                          (a * prev_std * transpose prev_std) +
                          (b * prev_q)) in
      Tensor.set q [t] new_q;
    done;
    
    let loss = Tensor.(- (sum (log (det q)))) in
    let grad_a, grad_b = Tensor.grad [a; b] loss in
    a <- Tensor.(a - Scalar.f learning_rate * grad_a);
    b <- Tensor.(b - Scalar.f learning_rate * grad_b);
  done;

  (univariate_models, (Tensor.to_float0_exn a, Tensor.to_float0_exn b))

let forecast_dcc_garch returns univariate_models dcc_params horizon =
  let n_assets = Array.length univariate_models in
  let a, b = dcc_params in
  let univariate_forecasts = Array.init n_assets (fun i ->
    let asset_returns = Tensor.select returns ~dim:1 ~index:i in
    let omega, alpha, _, beta = univariate_models.(i) in
    forecast_volatility GARCH (omega, alpha, 0.0, beta) asset_returns horizon
  ) in
  
  let q_bar = Tensor.(matmul (transpose returns) returns) in
  let forecasted_corr = Tensor.zeros [horizon; n_assets; n_assets] in
  let last_q = Tensor.get q_bar [Tensor.shape q_bar |> List.hd - 1] in
  
  for h = 0 to horizon - 1 do
    let new_q = Tensor.((Scalar.f (1.0 -. a -. b) * q_bar) + (Scalar.f a * last_q) + (Scalar.f b * last_q)) in
    let q_diag = Tensor.diag (sqrt (Tensor.diag new_q)) in
    let corr = Tensor.(matmul (matmul (inverse q_diag) new_q) (inverse q_diag)) in
    Tensor.set forecasted_corr [h] corr;
  done;

  (univariate_forecasts, forecasted_corr)