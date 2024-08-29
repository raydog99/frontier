open Base
open Torch

let generate_black_scholes_paths initial_price num_paths num_steps dt r sigma =
  let shape = [num_paths; num_steps + 1; 1] in
  let dw = Tensor.normal shape ~mean:0. ~std:(Float.sqrt dt) in
  let paths = Tensor.zeros shape in
  let _ = Tensor.index_put_ paths [Some (Tensor.tensor [0]); None; None] initial_price in
  let increments = Tensor.((float (r -. 0.5 *. sigma ** 2.) * float dt) + (float sigma * dw)) in
  let _ = Tensor.(paths.narrow 1 1 num_steps *= exp increments) in
  paths

let european_call_payoff strike prices =
  Tensor.max (Tensor.sub prices (Tensor.full_like prices strike)) (Tensor.zeros_like prices)

let european_put_payoff strike prices =
  Tensor.max (Tensor.sub (Tensor.full_like prices strike) prices) (Tensor.zeros_like prices)

let asian_call_payoff strike average_prices =
  Tensor.max (Tensor.sub average_prices (Tensor.full_like average_prices strike)) (Tensor.zeros_like average_prices)

let asian_put_payoff strike average_prices =
  Tensor.max (Tensor.sub (Tensor.full_like average_prices strike) average_prices) (Tensor.zeros_like average_prices)

let black_scholes_analytical_price ~spot ~strike ~risk_free_rate ~volatility ~maturity ~option_type =
  let d1 = (Float.log (spot /. strike) +. (risk_free_rate +. 0.5 *. volatility ** 2.) *. maturity) /. (volatility *. Float.sqrt maturity) in
  let d2 = d1 -. volatility *. Float.sqrt maturity in
  let nd1 = Stdlib.Float.erfc (-. d1 /. Float.sqrt 2.) /. 2. in
  let nd2 = Stdlib.Float.erfc (-. d2 /. Float.sqrt 2.) /. 2. in
  match option_type with
  | `Call -> spot *. nd1 -. strike *. Float.exp (-. risk_free_rate *. maturity) *. nd2
  | `Put -> strike *. Float.exp (-. risk_free_rate *. maturity) *. (1. -. nd2) -. spot *. (1. -. nd1)