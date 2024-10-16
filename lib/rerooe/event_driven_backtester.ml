open Lwt

type event =
  | PriceUpdate of { symbol: string; price: float; volume: float }
  | OrderExecution of { symbol: string; quantity: int; price: float }
  | RebalancePortfolio
  | RiskCheck

type t = {
  mutable current_time: float;
  mutable portfolio: Portfolio.t;
  mutable events: (float * event) list;
  risk_management: Risk_management.t option;
  constrained_optimizer: Constrained_optimizer.t option;
}

let create initial_portfolio =
  {
    current_time = 0.;
    portfolio = initial_portfolio;
    events = [];
    risk_management = None;
    constrained_optimizer = None;
  }

let add_event backtester time event =
  backtester.events <- (time, event) :: backtester.events;
  backtester.events <- List.sort (fun (t1, _) (t2, _) -> compare t1 t2) backtester.events

let process_event backtester event =
  match event with
  | PriceUpdate { symbol; price; volume } ->
      Portfolio.update_asset_price backtester.portfolio symbol price volume
  | OrderExecution { symbol; quantity; price } ->
      Portfolio.execute_order backtester.portfolio symbol quantity price
  | RebalancePortfolio ->
      (match backtester.constrained_optimizer with
       | Some optimizer -> Constrained_optimizer.optimize backtester.portfolio optimizer
       | None -> ())
  | RiskCheck ->
      (match backtester.risk_management with
       | Some risk_mgmt -> Risk_management.adjust_position_advanced backtester.portfolio risk_mgmt
       | None -> ())

let run backtester end_time =
  let rec process_events () =
    match backtester.events with
    | (time, event) :: rest when time <= end_time ->
        backtester.current_time <- time;
        backtester.events <- rest;
        process_event backtester event;
        process_events ()
    | _ -> ()
  in
  process_events ();
  backtester.current_time <- end_time

let get_results backtester =
  Portfolio.get_performance_summary backtester.portfolio

let backtest_strategy strategy initial_portfolio historical_data risk_management constrained_optimizer =
  let backtester = create initial_portfolio in
  backtester.risk_management <- risk_management;
  backtester.constrained_optimizer <- constrained_optimizer;

  (* Add events based on historical data *)
  Array.iteri (fun i (time, price, volume) ->
    add_event backtester time (PriceUpdate { symbol = "ASSET1"; price; volume });
    if i mod 10 = 0 then  (* Rebalance every 10 time steps *)
      add_event backtester time RebalancePortfolio;
    if i mod 5 = 0 then  (* Check risk every 5 time steps *)
      add_event backtester time RiskCheck;
    
    (* Execute strategy *)
    let action = strategy backtester.portfolio price in
    match action with
    | Some (symbol, quantity) ->
        add_event backtester time (OrderExecution { symbol; quantity; price })
    | None -> ()
  ) historical_data;

  let end_time = let (t, _, _) = historical_data.(Array.length historical_data - 1) in t in
  run backtester end_time;
  get_results backtester