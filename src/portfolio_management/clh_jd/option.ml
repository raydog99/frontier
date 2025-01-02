open Torch
open Utils

type option_type = Call | Put

type t = {
  option_type : option_type;
  strike : float;
  maturity : float;
}

let create ~option_type ~strike ~maturity =
  { option_type; strike; maturity }

let black_scholes_price option asset_price rate volatility time_to_maturity =
  let open Tensor in
  let d1 = (log (asset_price / float option.strike) + 
            (rate + (volatility ** float 2.) / float 2.) * time_to_maturity) /
           (volatility * sqrt time_to_maturity) in
  let d2 = d1 - volatility * sqrt time_to_maturity in
  
  match option.option_type with
  | Call ->
    asset_price * Utils.normal_cdf d1 - 
    float option.strike * exp (-rate * time_to_maturity) * Utils.normal_cdf d2
  | Put ->
    float option.strike * exp (-rate * time_to_maturity) * Utils.normal_cdf (-d2) -
    asset_price * Utils.normal_cdf (-d1)

let black_scholes_delta option asset_price rate volatility time_to_maturity =
  let open Tensor in
  let d1 = (log (asset_price / float option.strike) + 
            (rate + (volatility ** float 2.) / float 2.) * time_to_maturity) /
           (volatility * sqrt time_to_maturity) in
  
  match option.option_type with
  | Call -> Utils.normal_cdf d1
  | Put -> Utils.normal_cdf d1 - float 1.