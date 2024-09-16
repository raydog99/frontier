open Torch

type t = {
  initial_holdings: float;
  liquidation_time: float;
  num_intervals: int;
}

let create initial_holdings liquidation_time num_intervals =
  { initial_holdings; liquidation_time; num_intervals }

let get_trading_trajectory strategy =
  let { initial_holdings; liquidation_time; num_intervals } = strategy in
  let interval_length = liquidation_time /. float_of_int num_intervals in
  Tensor.arange ~start:0 ~end_:(float_of_int (num_intervals + 1)) ~options:(T Float)
  |> Tensor.mul_scalar (interval_length /. liquidation_time)
  |> Tensor.mul_scalar (-.initial_holdings)
  |> Tensor.add_scalar initial_holdings

let get_trade_list strategy =
  let trajectory = get_trading_trajectory strategy in
  Tensor.(trajectory - (trajectory.roll ~shifts:1 ~dims:[0]))