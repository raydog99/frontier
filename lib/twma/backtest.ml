open Torch
open Multi_asset_twma

type t = {
  config: Config.t;
  multi_asset_twma: Multi_asset_twma.t;
  mutable portfolio_value: float;
  mutable returns: float list;
}

let create config multi_asset_twma =
  {
    config;
    multi_asset_twma;
    portfolio_value = config.initial_cash;
    returns = [];
  }

let apply_transaction_costs t trade_value =
  trade_value *. (1. -. t.config.transaction_cost)

let apply_slippage t price direction =
  match direction with
  | Order.Buy -> price *. (1. +. t.config.slippage)
  | Order.Sell -> price *. (1. -. t.config.slippage)

let run t price_data =
  Array.iteri (fun day prices ->
    let current_weights = Multi_asset_twma.get_weights t.multi_asset_twma in
    let new_weights = Tensor.to_float1_exn (Multi_asset_twma.trade t.multi_asset_twma prices) in
    
    (* Calculate trades *)
    let trades = Array.map2 (fun cw nw -> nw -. cw) (Tensor.to_float1_exn current_weights) new_weights in
    
    (* Apply transaction costs and slippage *)
    let adjusted_portfolio_value = 
      Array.fold_left2 (fun acc trade price ->
        let trade_value = abs_float (trade *. t.portfolio_value) in
        let direction = if trade > 0. then Order.Buy else Order.Sell in
        let slippage_price = apply_slippage t price direction in
        let after_costs = apply_transaction_costs t trade_value in
        acc +. after_costs /. slippage_price
      ) t.portfolio_value trades prices
    in
    
    let daily_return = (adjusted_portfolio_value -. t.portfolio_value) /. t.portfolio_value in
    t.returns <- daily_return :: t.returns;
    t.portfolio_value <- adjusted_portfolio_value;

    Logger.info (Printf.sprintf "Day %d: Portfolio Value: %.2f, Daily Return: %.2f%%" 
                   day t.portfolio_value (daily_return *. 100.))
  ) price_data;
  List.rev t.returns