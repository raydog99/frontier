open Torch
open Types

let optimize config data =
  let returns = Portfolio.calculate_returns { assets = []; weights = Tensor.zeros [1; config.Config.n_assets] } in
  let covariance = Rmt.estimate_covariance data config.covariance_method in
  
  match config.optimization_method with
  | `Markowitz -> Portfolio.markowitz_optimize returns covariance config.target_return
  | `NCO -> Nco.optimize data