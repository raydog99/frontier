open Torch
open Asset
open Option

type hedging_strategy = {
  asset : Asset.t;
  option : Option.t;
  transaction_cost : float;
  time_steps : int;
  dt : float;
}

let create ~asset ~option ~transaction_cost ~time_steps ~dt =
  { asset; option; transaction_cost; time_steps; dt }

let conditional_expectation_f strategy asset_price time =
  let open Tensor in
  let time_to_maturity = float strategy.option.maturity - time in
  Option.black_scholes_price strategy.option asset_price (float 0.) (float strategy.asset.volatility) time_to_maturity

let conditional_expectation_s_f strategy asset_price time =
  let open Tensor in
  let time_to_maturity = float strategy.option.maturity - time in
  asset_price * Option.black_scholes_delta strategy.option asset_price (float 0.) (float strategy.asset.volatility) time_to_maturity

let calculate_u strategy asset_price portfolio_value position time =
  let open Tensor in
  let s_hat = asset_price * exp ((float strategy.asset.drift + float strategy.asset.jump_intensity) * float strategy.dt) in
  let s_hat_squared = s_hat ** float 2. in
  
  (portfolio_value - position * asset_price) * s_hat -
  conditional_expectation_s_f strategy asset_price time +
  position * s_hat_squared

let clh_strategy strategy asset_price portfolio_value position time =
  let open Tensor in
  let u = calculate_u strategy asset_price portfolio_value position time in
  let l_n = (portfolio_value - position * asset_price) * asset_price * 
            exp ((float strategy.asset.drift + float strategy.asset.jump_intensity) * float strategy.dt) +
            position * (asset_price ** float 2.) * 
            exp ((float 2. * float strategy.asset.drift + float strategy.asset.volatility ** float 2. + 
                  float 2. * float strategy.asset.jump_intensity + 
                  log (float 1. + float strategy.asset.jump_size) ** float 2. * 
                  float strategy.asset.jump_intensity) * float strategy.dt) in
  let l_call = conditional_expectation_s_f strategy asset_price time in
  
  if l_n <= l_call then
    position
  else
    let delta_long = position + (l_n - l_call) / 
                     (asset_price ** float 2. * 
                      exp ((float 2. * float strategy.asset.drift + float strategy.asset.volatility ** float 2. + 
                            float 2. * float strategy.asset.jump_intensity + 
                            log (float 1. + float strategy.asset.jump_size) ** float 2. * 
                            float strategy.asset.jump_intensity) * float strategy.dt)) in
    let delta_short = position - (l_n - l_call) / 
                      (asset_price ** float 2. * 
                       exp ((float 2. * float strategy.asset.drift + float strategy.asset.volatility ** float 2. + 
                             float 2. * float strategy.asset.jump_intensity + 
                             log (float 1. + float strategy.asset.jump_size) ** float 2. * 
                             float strategy.asset.jump_intensity) * float strategy.dt)) in
    if u > float 0. then delta_long else delta_short

let cmh_strategy strategy asset_price portfolio_value position time =
  let open Tensor in
  let u = calculate_u strategy asset_price portfolio_value position time in
  let l_n = (portfolio_value - position * asset_price) * asset_price * 
            exp ((float strategy.asset.drift + float strategy.asset.jump_intensity) * float strategy.dt) +
            position * (asset_price ** float 2.) * 
            exp ((float 2. * float strategy.asset.drift + float strategy.asset.volatility ** float 2. + 
                  float 2. * float strategy.asset.jump_intensity + 
                  log (float 1. + float strategy.asset.jump_size) ** float 2. * 
                  float strategy.asset.jump_intensity) * float strategy.dt) in
  let l_call = conditional_expectation_s_f strategy asset_price time in
  
  if l_n <= l_call then
    position
  else
    let delta = (l_n - l_call) / 
                (asset_price * 
                 exp ((float strategy.asset.drift + float strategy.asset.jump_intensity) * float strategy.dt)) in
    if u > float 0. then
      position + delta
    else
      position - delta

type hedging_method = CLH | CMH

let hedge strategy asset_price portfolio_value position time method =
  match method with
  | CLH -> clh_strategy strategy asset_price portfolio_value position time
  | CMH -> cmh_strategy strategy asset_price portfolio_value position time