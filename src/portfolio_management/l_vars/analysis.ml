open Torch

let calculate_implementation_shortfall trajectory prices initial_price =
  let trade_list = Tensor.(trajectory - (trajectory.roll ~shifts:1 ~dims:[0])) in
  let execution_prices = Tensor.(prices * trade_list) in
  let total_traded = Tensor.sum trade_list in
  let vwap = Tensor.(sum execution_prices / total_traded) in
  Tensor.to_float0_exn Tensor.(vwap - f initial_price)

let calculate_vwap trajectory prices =
  let trade_list = Tensor.(trajectory - (trajectory.roll ~shifts:1 ~dims:[0])) in
  let execution_prices = Tensor.(prices * trade_list) in
  let total_traded = Tensor.sum trade_list in
  Tensor.to_float0_exn Tensor.(sum execution_prices / total_traded)

let calculate_participation_rate trajectory daily_volume =
  let trade_list = Tensor.(trajectory - (trajectory.roll ~shifts:1 ~dims:[0])) in
  Tensor.to_float1 Tensor.(abs trade_list / f daily_volume)