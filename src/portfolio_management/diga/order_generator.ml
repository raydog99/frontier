open Torch

type order = {
  timestamp: float;
  price: float;
  volume: int;
  is_buy: bool;
}

type t = {
  initial_price: float;
  risk_aversion: float;
  fundamental_weight: float;
  chartist_weight: float;
  noise_weight: float;
}

let create ~initial_price ~risk_aversion ~fundamental_weight ~chartist_weight ~noise_weight =
  { initial_price; risk_aversion; fundamental_weight; chartist_weight; noise_weight }

let estimate_return t ~fundamental ~historical_return =
  let noise = Tensor.randn [] in
  Tensor.(
    (t.fundamental_weight * fundamental +
     t.chartist_weight * historical_return +
     t.noise_weight * noise) /
    (t.fundamental_weight + t.chartist_weight + t.noise_weight)
  )

let generate_order t ~current_price ~estimated_return ~volatility ~timestamp =
  let estimated_price = Tensor.(current_price * exp estimated_return) in
  let demand = Tensor.((log (estimated_price / current_price)) / (t.risk_aversion * volatility * current_price)) in
  let order_price = Tensor.(current_price * exp (Tensor.uniform [] ~from:0. ~to:1. * estimated_return)) in
  let volume = Tensor.(abs (demand - Tensor.uniform [] ~from:(-1.) ~to:1.)) |> Tensor.to_int0_exn in
  let is_buy = Tensor.(demand > Tensor.zeros_like demand) |> Tensor.to_int0_exn |> (fun x -> x = 1) in
  { timestamp; price = Tensor.to_float0_exn order_price; volume; is_buy }

let generate_orders t market_states =
  let seq_length = Tensor.shape market_states |> List.nth (-1) in
  let current_price = ref t.initial_price in
  let historical_returns = ref [] in
  List.init seq_length (fun i ->
    let fundamental = Tensor.slice market_states ~dim:1 ~start:0 ~end_:1 |> Tensor.select ~dim:1 ~index:0 |> Tensor.select ~dim:0 ~index:i in
    let arrival_rate = Tensor.slice market_states ~dim:1 ~start:1 ~end_:2 |> Tensor.select ~dim:1 ~index:0 |> Tensor.select ~dim:0 ~index:i in
    let historical_return = match !historical_returns with
      | [] -> Tensor.zeros []
      | returns -> Tensor.of_float1 returns |> Tensor.mean
    in
    let estimated_return = estimate_return t ~fundamental ~historical_return in
    let volatility = Tensor.std (Tensor.of_float1 !historical_returns) in
    let timestamp = float_of_int i in
    let order = generate_order t ~current_price:!current_price ~estimated_return ~volatility ~timestamp in
    current_price := if order.is_buy then !current_price *. 1.0001 else !current_price *. 0.9999;
    historical_returns := (Tensor.log (Tensor.of_float1 [!current_price /. t.initial_price])) :: !historical_returns;
    if List.length !historical_returns > 100 then
      historical_returns := List.tl !historical_returns;
    order
  )