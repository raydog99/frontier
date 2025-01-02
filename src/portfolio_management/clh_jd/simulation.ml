open Torch
open Asset
open Hedging

let simulate_hedging strategy num_simulations method =
  let open Tensor in
  
  let asset_prices = Asset.simulate_price strategy.Hedging.asset 
                       ~time_steps:strategy.Hedging.time_steps 
                       ~dt:strategy.Hedging.dt in
  
  let portfolio_values = zeros [strategy.Hedging.time_steps + 1; num_simulations] in
  let positions = zeros [strategy.Hedging.time_steps + 1; num_simulations] in
  
  for sim = 0 to num_simulations - 1 do
    let initial_price = asset_prices.[[0]] in
    let initial_portfolio_value = Hedging.conditional_expectation_f strategy initial_price (float 0.) in
    let initial_position = Hedging.conditional_expectation_s_f strategy initial_price (float 0.) / initial_price in
    
    portfolio_values.[[0; sim]] <- initial_portfolio_value;
    positions.[[0; sim]] <- initial_position;
    
    for t = 1 to strategy.Hedging.time_steps do
      let prev_position = positions.[[t-1; sim]] in
      let prev_portfolio_value = portfolio_values.[[t-1; sim]] in
      let current_price = asset_prices.[[t]] in
      let time = float t *. strategy.Hedging.dt in
      
      let new_position = Hedging.hedge strategy current_price prev_portfolio_value prev_position time method in
      let transaction_cost = strategy.Hedging.transaction_cost *. current_price.to_float0 *. abs_float (new_position.to_float0 -. prev_position.to_float0) in
      
      let new_portfolio_value = prev_portfolio_value + 
                                prev_position * (current_price - asset_prices.[[t-1]]) - 
                                float transaction_cost in
      
      portfolio_values.[[t; sim]] <- new_portfolio_value;
      positions.[[t; sim]] <- new_position;
    done;
  done;
  
  (asset_prices, portfolio_values, positions)

let calculate_hedging_error asset_prices portfolio_values option strike =
  let open Tensor in
  let final_asset_prices = asset_prices.[-1] in
  let final_portfolio_values = portfolio_values.[-1] in
  let option_payoffs = max (final_asset_prices - float strike) (float 0.) in
  abs (final_portfolio_values - option_payoffs)

let calculate_mean_hedging_error asset_prices portfolio_values option =
  let open Tensor in
  let hedging_errors = calculate_hedging_error asset_prices portfolio_values option option.Option.strike in
  mean hedging_errors

let calculate_var_hedging_error asset_prices portfolio_values option confidence_level =
  let open Tensor in
  let hedging_errors = calculate_hedging_error asset_prices portfolio_values option option.Option.strike in
  let sorted_errors = sort hedging_errors in
  let index = int_of_float (float (numel sorted_errors) *. confidence_level) in
  sorted_errors.[[index]]

let calculate_sharpe_ratio portfolio_values dt =
  let open Tensor in
  let returns = log (portfolio_values.[-1] / portfolio_values.[0]) in
  let mean_return = mean returns in
  let std_return = std returns ~dim:[0] ~unbiased:true ~keepdim:false in
  (mean_return - float (Stdlib.log 1.0)) / std_return * (float (sqrt (1. /. dt)))

let calculate_maximum_drawdown portfolio_values =
  let open Tensor in
  let cummax = cummax portfolio_values ~dim:0 |> fst in
  let drawdowns = (cummax - portfolio_values) / cummax in
  max drawdowns

let print_advanced_statistics asset_prices portfolio_values positions option dt =
  let open Tensor in
  let hedging_errors = calculate_hedging_error asset_prices portfolio_values option option.Option.strike in
  let mean_error = mean hedging_errors in
  let std_error = std hedging_errors ~dim:[0] ~unbiased:true ~keepdim:false in
  let var_95 = calculate_var_hedging_error asset_prices portfolio_values option 0.95 in
  let var_99 = calculate_var_hedging_error asset_prices portfolio_values option 0.99 in
  let sharpe_ratio = calculate_sharpe_ratio portfolio_values dt in
  let max_drawdown = calculate_maximum_drawdown portfolio_values in
  
  Printf.printf "Advanced Statistics:\n";
  Printf.printf "Mean Hedging Error: %.4f\n" (mean_error.to_float0);
  Printf.printf "Std Dev of Hedging Error: %.4f\n" (std_error.to_float0);
  Printf.printf "95%% VaR of Hedging Error: %.4f\n" (var_95.to_float0);
  Printf.printf "99%% VaR of Hedging Error: %.4f\n" (var_99.to_float0);
  Printf.printf "Sharpe Ratio: %.4f\n" (sharpe_ratio.to_float0);
  Printf.printf "Maximum Drawdown: %.4f\n" (max_drawdown.to_float0)

let compare_hedging_methods strategy num_simulations =
  let (clh_asset_prices, clh_portfolio_values, clh_positions) = simulate_hedging strategy num_simulations Hedging.CLH in
  let (cmh_asset_prices, cmh_portfolio_values, cmh_positions) = simulate_hedging strategy num_simulations Hedging.CMH in
  
  Printf.printf "CLH Results:\n";
  print_advanced_statistics clh_asset_prices clh_portfolio_values clh_positions strategy.Hedging.option strategy.Hedging.dt;
  
  Printf.printf "\nCMH Results:\n";
  print_advanced_statistics cmh_asset_prices cmh_portfolio_values cmh_positions strategy.Hedging.option strategy.Hedging.dt;
  
  let clh_errors = calculate_hedging_error clh_asset_prices clh_portfolio_values strategy.Hedging.option strategy.Hedging.option.Option.strike in
  let cmh_errors = calculate_hedging_error cmh_asset_prices cmh_portfolio_values strategy.Hedging.option strategy.Hedging.option.Option.strike in
  
  let clh_mean_error = Tensor.mean clh_errors in
  let cmh_mean_error = Tensor.mean cmh_errors in
  
  Printf.printf "\nComparison:\n";
  Printf.printf "CLH Mean Hedging Error: %.4f\n" (clh_mean_error.to_float0);
  Printf.printf "CMH Mean Hedging Error: %.4f\n" (cmh_mean_error.to_float0);
  Printf.printf "Difference (CLH - CMH): %.4f\n" (Tensor.(clh_mean_error - cmh_mean_error).to_float0)