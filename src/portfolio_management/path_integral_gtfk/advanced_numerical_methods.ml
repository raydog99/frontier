open Base
open Torch

let finite_difference_greeks ~option_pricing_fn ~spot ~strike ~risk_free_rate ~volatility ~maturity ~option_type =
  let h_spot = 1e-4 *. spot in
  let h_vol = 1e-4 *. volatility in
  let h_time = 1e-4 *. maturity in

  let v = option_pricing_fn ~spot in
  let v_up = option_pricing_fn ~spot:(spot +. h_spot) in
  let v_down = option_pricing_fn ~spot:(spot -. h_spot) in
  let v_vol_up = option_pricing_fn ~spot ~volatility:(volatility +. h_vol) in
  let v_vol_down = option_pricing_fn ~spot ~volatility:(volatility -. h_vol) in
  let v_time_down = option_pricing_fn ~spot ~maturity:(maturity -. h_time) in

  let delta = (v_up -. v_down) /. (2. *. h_spot) in
  let gamma = (v_up -. 2. *. v +. v_down) /. (h_spot *. h_spot) in
  let vega = (v_vol_up -. v_vol_down) /. (2. *. h_vol) in
  let theta = (v -. v_time_down) /. h_time in

  (delta, gamma, vega, theta)