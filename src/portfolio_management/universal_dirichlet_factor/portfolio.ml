open Torch

let generate_factor_market num_assets num_timesteps =
  let factor = Tensor.randn [num_timesteps] in
  let asset_exposures = Tensor.randn [num_assets] in
  let asset_returns = Tensor.mm (Tensor.unsqueeze factor 1) (Tensor.unsqueeze asset_exposures 0) in
  asset_returns

let generate_multi_factor_market num_assets num_factors num_timesteps =
  let factors = Tensor.randn [num_timesteps; num_factors] in
  let asset_exposures = Tensor.randn [num_factors; num_assets] in
  let asset_returns = Tensor.mm factors asset_exposures in
  asset_returns

let create_crp num_assets =
  let weights = Tensor.ones [num_assets] in
  Tensor.div_scalar weights (Tensor.sum weights)

let sample_dirichlet alpha =
  let gamma_samples = Tensor.map (fun a -> Tensor.gamma_1 a 1.0) alpha in
  let sum = Tensor.sum gamma_samples in
  Tensor.div gamma_samples sum

let create_dfp num_factors concentration =
  let alpha = Tensor.full [num_factors] concentration in
  sample_dirichlet alpha

let portfolio_return weights asset_returns =
  Tensor.sum (Tensor.mul weights asset_returns)

let growth_rate weights asset_returns =
  let returns = portfolio_return weights asset_returns in
  Tensor.log1p returns |> Tensor.mean

let simulate_dfp num_assets num_factors num_timesteps concentration =
  let asset_returns = generate_multi_factor_market num_assets num_factors num_timesteps in
  let dfp_weights = create_dfp num_factors concentration in
  let factor_returns = Tensor.mm asset_returns (Tensor.transpose dfp_weights 0 1) in
  let growth = growth_rate dfp_weights factor_returns in
  growth

let monte_carlo_dfp num_assets num_factors num_timesteps concentration num_simulations =
  let simulations = List.init num_simulations (fun _ -> 
    simulate_dfp num_assets num_factors num_timesteps concentration
  ) in
  let results = Tensor.of_float1 simulations in
  let mean_growth = Tensor.mean results in
  let std_growth = Tensor.std results in
  (mean_growth, std_growth)

let approximate_target_portfolio target_weights num_factors concentration num_samples =
  let alpha = Tensor.full [num_factors] concentration in
  let samples = Tensor.stack (List.init num_samples (fun _ -> sample_dirichlet alpha)) 0 in
  let best_approx = Tensor.mm samples (Tensor.unsqueeze target_weights 1) in
  let best_idx = Tensor.argmax (Tensor.squeeze best_approx 1) 0 in
  Tensor.select samples 0 best_idx

let tracking_error approx_weights target_weights asset_returns =
  let approx_returns = portfolio_return approx_weights asset_returns in
  let target_returns = portfolio_return target_weights asset_returns in
  let diff = Tensor.sub approx_returns target_returns in
  Tensor.std diff

let sharpe_ratio weights asset_returns risk_free_rate =
  let returns = portfolio_return weights asset_returns in
  let excess_returns = Tensor.sub returns risk_free_rate in
  let mean_excess_return = Tensor.mean excess_returns in
  let std_dev = Tensor.std excess_returns in
  Tensor.div mean_excess_return std_dev

let maximum_drawdown weights asset_returns =
  let cumulative_returns = Tensor.cumprod (Tensor.add (portfolio_return weights asset_returns) 1.0) 0 in
  let running_max = Tensor.cummax cumulative_returns 0 |> fst in
  let drawdowns = Tensor.div cumulative_returns running_max |> Tensor.sub_scalar 1.0 |> Tensor.neg in
  Tensor.max drawdowns

let moving_average tensor window =
  let kernel = Tensor.ones [1; 1; window] |> Tensor.div_scalar (float window) in
  Tensor.conv1d (Tensor.unsqueeze tensor 0) kernel 1 ~padding:((window - 1) / 2) |> Tensor.squeeze ~dim:0

let compare_dfp_to_equal_weight num_assets num_factors num_timesteps concentration num_simulations =
  let equal_weight = create_crp num_assets in
  let simulations = List.init num_simulations (fun _ ->
    let asset_returns = generate_multi_factor_market num_assets num_factors num_timesteps in
    let dfp_weights = create_dfp num_factors concentration in
    let factor_returns = Tensor.mm asset_returns (Tensor.transpose dfp_weights 0 1) in
    let dfp_growth = growth_rate dfp_weights factor_returns in
    let equal_weight_growth = growth_rate equal_weight asset_returns in
    (dfp_growth, equal_weight_growth)
  ) in
  let dfp_results, equal_weight_results = List.split simulations in
  let dfp_tensor = Tensor.of_float1 dfp_results in
  let equal_weight_tensor = Tensor.of_float1 equal_weight_results in
  (Tensor.mean dfp_tensor, Tensor.std dfp_tensor, 
   Tensor.mean equal_weight_tensor, Tensor.std equal_weight_tensor)

let confidence_interval mean std_dev confidence_level num_samples =
  let z_score = match confidence_level with
    | 0.90 -> 1.645
    | 0.95 -> 1.96
    | 0.99 -> 2.576
    | _ -> failwith "Unsupported confidence level" in
  let margin_of_error = z_score *. (Tensor.item std_dev) /. sqrt (float num_samples) in
  (Tensor.item mean -. margin_of_error, Tensor.item mean +. margin_of_error)

let rebalance_portfolio current_weights target_weights transaction_cost =
  let diff = Tensor.sub target_weights current_weights in
  let total_change = Tensor.sum (Tensor.abs diff) in
  let cost = Tensor.mul_scalar total_change transaction_cost in
  let new_weights = Tensor.add current_weights diff in
  let normalized_weights = Tensor.div new_weights (Tensor.sum new_weights) in
  (normalized_weights, cost)

let backtest strategy initial_weights asset_returns rebalance_frequency transaction_cost =
  let num_periods = Tensor.size asset_returns 0 in
  let num_assets = Tensor.size asset_returns 1 in
  
  let rec simulate period weights portfolio_value results =
    if period >= num_periods then
      List.rev results
    else
      let period_returns = Tensor.select asset_returns period 0 in
      let portfolio_return = Tensor.sum (Tensor.mul weights period_returns) in
      let new_portfolio_value = portfolio_value *. (1.0 +. Tensor.item portfolio_return) in
      
      let (new_weights, rebalance_cost) =
        if period mod rebalance_frequency = 0 then
          let target_weights = strategy num_assets in
          rebalance_portfolio weights target_weights transaction_cost
        else
          (weights, Tensor.of_float0 0.0)
      in
      
      let actual_portfolio_value = new_portfolio_value *. (1.0 -. Tensor.item rebalance_cost) in
      simulate (period + 1) new_weights actual_portfolio_value 
        ((actual_portfolio_value, new_weights) :: results)
  in
  
  simulate 0 initial_weights 1.0 []

let annualized_return final_value initial_value num_years =
  let total_return = (final_value /. initial_value) ** (1.0 /. num_years) -. 1.0 in
  total_return *. 100.0

let annualized_volatility returns num_periods_per_year =
  let std_dev = Tensor.std returns in
  Tensor.item std_dev *. sqrt (float num_periods_per_year) *. 100.0