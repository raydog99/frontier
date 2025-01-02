open Torch
open Types

let calculate_returns portfolio =
  let asset_returns = List.map (fun asset ->
    let prices = asset.prices in
    let log_returns = Tensor.(log (div (slice prices [-1] []) (slice prices [0] [-1]))) in
    log_returns
  ) portfolio.assets in
  Tensor.stack asset_returns

let calculate_expected_return portfolio =
  let returns = calculate_returns portfolio in
  Tensor.mm portfolio.weights (Tensor.mean returns ~dim:[0] ~keepdim:true)

let calculate_risk portfolio covariance =
  let risk_squared = Tensor.mm (Tensor.mm portfolio.weights covariance) (Tensor.transpose portfolio.weights 0 1) in
  Tensor.item risk_squared |> sqrt

let markowitz_optimize returns covariance target_return =
  let n = Tensor.size returns 1 in
  let ones = Tensor.ones [1; n] in
  let inv_cov = Tensor.inverse covariance in
  let a = Tensor.mm (Tensor.mm ones inv_cov) (Tensor.transpose returns 0 1) |> Tensor.item in
  let b = Tensor.mm (Tensor.mm returns inv_cov) (Tensor.transpose returns 0 1) |> Tensor.item in
  let c = Tensor.mm (Tensor.mm ones inv_cov) (Tensor.transpose ones 0 1) |> Tensor.item in
  let lambda = (c *. target_return -. a) /. (b *. c -. a *. a) in
  let gamma = (b -. a *. target_return) /. (b *. c -. a *. a) in
  Tensor.add 
    (Tensor.mul_scalar (Tensor.mm inv_cov (Tensor.transpose returns 0 1)) lambda)
    (Tensor.mul_scalar (Tensor.mm inv_cov (Tensor.transpose ones 0 1)) gamma)

let create_portfolio assets weights =
  { assets; weights }