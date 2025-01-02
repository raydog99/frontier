open Torch
open Types

let calculate_average_strategy strategies =
  Tensor.mean (Tensor.stack strategies ~dim:0)

let calculate_clearing_error average_strategy =
  Tensor.to_float0_exn (Tensor.norm average_strategy)

let is_market_cleared average_strategy epsilon =
  calculate_clearing_error average_strategy < epsilon

let update_market_clearing_state strategies =
  let average_strategy = calculate_average_strategy strategies in
  let clearing_error = calculate_clearing_error average_strategy in
  { average_strategy; clearing_error }

let adjust_risk_premium risk_premium market_clearing_state params =
  Tensor.(sub risk_premium (mul (float params.gamma) market_clearing_state.average_strategy))