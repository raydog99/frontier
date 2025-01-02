open Torch

let value_at_risk model portfolio confidence_level num_simulations =
  let returns = List.init num_simulations (fun _ ->
    Nmvm.sample model |> fun r -> Tensor.(dot portfolio r |> to_float0)
  ) in
  let sorted_returns = List.sort compare returns in
  List.nth sorted_returns (int_of_float (float_of_int num_simulations *. (1. -. confidence_level)))

let conditional_value_at_risk model portfolio confidence_level num_simulations =
  let returns = List.init num_simulations (fun _ ->
    Nmvm.sample model |> fun r -> Tensor.(dot portfolio r |> to_float0)
  ) in
  let sorted_returns = List.sort compare returns in
  let var_index = int_of_float (float_of_int num_simulations *. (1. -. confidence_level)) in
  let tail_returns = List.filteri (fun i _ -> i < var_index) sorted_returns in
  List.fold_left (+.) 0. tail_returns /. float_of_int (List.length tail_returns)

let expected_shortfall model portfolio confidence_level num_simulations =
  conditional_value_at_risk model portfolio confidence_level num_simulations

let stress_test model portfolio scenarios =
  List.map (fun (mu_shift, gamma_multiplier, volatility_multiplier) ->
    let stressed_model = Nmvm.create
      (Tensor.add model.Nmvm.mu mu_shift)
      (Tensor.mul model.Nmvm.gamma (Tensor.scalar_float gamma_multiplier))
      (Tensor.mul model.Nmvm.sigma (Tensor.scalar_float volatility_multiplier))
      model.Nmvm.z_dist
    in
    let returns = Nmvm.sample stressed_model in
    Tensor.(dot portfolio returns |> to_float0)
  ) scenarios

let portfolio_beta model portfolio market_portfolio =
  let portfolio_returns = Nmvm.generate_samples model 1000 |> List.map (fun r -> Tensor.(dot portfolio r |> to_float0)) in
  let market_returns = Nmvm.generate_samples model 1000 |> List.map (fun r -> Tensor.(dot market_portfolio r |> to_float0)) in
  let cov = Owl.Stats.covariance portfolio_returns market_returns in
  let market_var = Owl.Stats.var market_returns in
  cov /. market_var

let tracking_error model portfolio benchmark num_simulations =
  let portfolio_returns = List.init num_simulations (fun _ ->
    Nmvm.sample model |> fun r -> Tensor.(dot portfolio r |> to_float0)
  ) in
  let benchmark_returns = List.init num_simulations (fun _ ->
    Nmvm.sample model |> fun r -> Tensor.(dot benchmark r |> to_float0)
  ) in
  let excess_returns = List.map2 (-.) portfolio_returns benchmark_returns in
  Owl.Stats.std excess_returns