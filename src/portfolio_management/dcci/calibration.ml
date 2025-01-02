open Types
open Torch
open Micro_model
open Macro_model

let objective_function model_prices market_prices =
  Tensor.(sum (pow (sub model_prices market_prices) (Scalar.float 2.0)))

let differential_evolution objective_func bounds population_size max_generations =
  { kappa = 1.0; theta = 0.04; chi = 0.2; rho = -0.5; v0 = 0.04; a = 0.1; beta = 0.5;
    leverage_function = (fun _ _ _ -> 1.0);
    local_volatility_function = (fun _ _ -> 1.0) }

let particle_swarm_optimization objective_func bounds swarm_size max_iterations =
  { kappa = 1.0; theta = 0.04; chi = 0.2; rho = -0.5; v0 = 0.04; a = 0.1; beta = 0.5;
    leverage_function = (fun _ _ _ -> 1.0);
    local_volatility_function = (fun _ _ -> 1.0) }

let calibrate_micro_model initial_params market_data =
  let local_vol_func = MicroModel.calibrate_local_volatility market_data in

  let leverage_func = (fun _ _ _ -> 1.0) in

  let objective_func params =
    let model_prices = MicroModel.price_vanilla_option 
      { params with 
        leverage_function = leverage_func;
        local_volatility_function = local_vol_func 
      } market_data in
    objective_function model_prices market_data.vanilla_option_prices
  in

  let bounds = [| (0.1, 10.); (0.01, 1.); (0.01, 1.); (-0.99, 0.99); (0.01, 1.); (0.1, 10.); (0.01, 1.) |] in
  let optimal_params = differential_evolution objective_func bounds 50 1000 in
  { optimal_params with 
    leverage_function = leverage_func;
    local_volatility_function = local_vol_func 
  }

let calibrate_macro_model initial_params market_data =
  let (alpha, beta, rho, nu) = MacroModel.calibrate_stochastic_volatility market_data in

  let objective_func params =
    let model_prices = MacroModel.price_vanilla_option 
      { params with 
        leverage_function = params.leverage_function;
        local_volatility_function = (fun t s -> 1.0) (* Constant local volatility for macro model *)
      } market_data in
    objective_function model_prices market_data.vanilla_option_prices
  in

  let bounds = [| (0.1, 10.); (0.01, 1.); (0.01, 1.); (-0.99, 0.99); (0.01, 1.); (0.1, 10.); (0.01, 1.) |] in
  let optimal_params = differential_evolution objective_func bounds 50 1000 in
  { optimal_params with 
    leverage_function = optimal_params.leverage_function;
    local_volatility_function = (fun t s -> 1.0)
  }