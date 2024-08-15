open Types
open Torch
open Utils
open Numerical_methods

module MacroModel : MODEL = struct
  let simulate params market_data num_paths num_steps =
    let dt = market_data.futures_maturities.(1) -. market_data.futures_maturities.(0) in
    let sqrt_dt = sqrt dt in
    let paths = Tensor.zeros [num_paths; num_steps] in
    let variance_paths = Tensor.zeros [num_paths; num_steps] in

    for t = 1 to num_steps - 1 do
      let prev_index = Tensor.select paths 1 (t-1) in
      let prev_var = Tensor.select variance_paths 1 (t-1) in

      let dW_S = Tensor.randn [num_paths] in
      let dW_v = Tensor.(params.rho * dW_S + sqrt (Scalar.float (1. -. params.rho *. params.rho)) * randn [num_paths]) in

      let new_var = Tensor.(
        prev_var + 
        (params.kappa * (Scalar.float params.theta - prev_var)) * Scalar.float dt +
        (params.chi * sqrt prev_var * dW_v * Scalar.float sqrt_dt)
      ) in

      let vol = Tensor.(sqrt new_var * params.local_volatility_function t prev_index) in
      let new_index = Tensor.(
        prev_index * exp ((Scalar.float market_data.risk_free_rate - Scalar.float 0.5 * vol * vol) * Scalar.float dt + 
                          vol * dW_S * Scalar.float sqrt_dt)
      ) in

      Tensor.select_inplace paths 1 t new_index;
      Tensor.select_inplace variance_paths 1 t new_var
    done;
    paths

  let price_vanilla_option params market_data strike maturity option_type =
    let num_paths = 10000 in
    let num_steps = 100 in
    let paths = simulate params market_data num_paths num_steps in
    let final_prices = Tensor.select paths 1 (num_steps - 1) in
    let payoffs = match option_type with
      | Call -> Tensor.(max (final_prices - Scalar.float strike) (Scalar.float 0.))
      | Put -> Tensor.(max (Scalar.float strike - final_prices) (Scalar.float 0.))
    in
    let discount_factor = exp (-. market_data.risk_free_rate *. maturity) in
    Tensor.(mean payoffs * Scalar.float discount_factor)

  let price_path_dependent_option params market_data option num_paths num_steps =
    let paths = simulate params market_data num_paths num_steps in
    match option with
    | Autocallable contract -> price_autocallable params market_data contract paths
    | AthenaJet contract -> price_athena_jet params market_data contract paths
    | DailyBarrierKnockIn { barrier; maturity } -> price_daily_barrier_knock_in params market_data barrier maturity paths

  let price_autocallable params market_data contract paths =
    0.0

  let price_athena_jet params market_data contract paths =
    0.0

  let price_daily_barrier_knock_in params market_data barrier maturity paths =
    0.0

  let compute_delta params market_data option =
    0.0

  let compute_vega params market_data option =
    0.0

  let calibrate_stochastic_volatility market_data =
    fit_sabr_model market_data.index_price market_data.vanilla_option_prices market_data.vanilla_option_strikes market_data.vanilla_option_maturities

  let price_with_characteristic_function params market_data option_type strike maturity =
    let char_func = heston_characteristic_function params in
    fast_fourier_transform_option_pricing char_func market_data.index_price strike maturity
end