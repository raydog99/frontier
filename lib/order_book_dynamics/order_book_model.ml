open Printf

type order_type = Market | Limit | Cancel | Iceberg
type side = Buy | Sell

type order = {
  order_type: order_type;
  side: side;
  price: int;
  size: int;
  asset_id: int;
  hidden_size: int option;
}

type order_book = {
  bids: (int * int) list;
  asks: (int * int) list;
  asset_id: int;
  hidden_orders: (side * int * int) list;
}

type simulation_stats = {
  mutable num_market_orders: int;
  mutable num_limit_orders: int;
  mutable num_cancellations: int;
  mutable price_changes: int;
  mutable volume_traded: int;
  mutable mid_prices: (int * float) list;
}

type portfolio = {
  mutable cash: float;
  mutable positions: (int * int) list;
}

type market_impact = {
  temporary: float -> int -> float;
  permanent: float -> int -> float;
}

type execution_algo = {
  name: string;
  execute: order -> order_book list -> int -> order list;
}

type event =
  | OrderEvent of order
  | TradeEvent of order * int  (* order and executed price *)
  | MarketDataEvent of order_book

let run_simulation num_iterations initial_books strategies =
  let (final_books, stats, final_portfolios) = Simulation.run_event_driven num_iterations initial_books strategies in
  (final_books, stats, final_portfolios)

let analyze_results final_books stats final_portfolios strategies =
  List.iter2 (fun strategy portfolio ->
    let returns = List.map (fun (_, price) -> log (price /. List.hd stats.mid_prices |> snd)) (List.rev stats.mid_prices) in
    printf "Strategy: %s\n" strategy.TradingStrategy.name;
    printf "Final portfolio value: %.2f\n" (portfolio.cash +. List.fold_left (fun acc (asset_id, qty) ->
      let book = List.find (fun b -> b.asset_id = asset_id) final_books in
      acc +. float_of_int qty *. float_of_int (OrderBook.get_mid_price book)
    ) 0. portfolio.positions);
    printf "Sharpe ratio: %.4f\n" (Analysis.calculate_sharpe_ratio returns 0.);
    printf "Sortino ratio: %.4f\n" (Analysis.calculate_sortino_ratio returns 0.);
    printf "Maximum drawdown: %.4f\n" (Analysis.calculate_maximum_drawdown returns);
    printf "Value at Risk (95%%): %.4f\n" (Analysis.calculate_var returns 0.95);
    printf "Expected Shortfall (95%%): %.4f\n" (Analysis.calculate_expected_shortfall returns 0.95);
    printf "\n";

    Visualization.plot_price_series (List.map snd stats.mid_prices) (strategy.TradingStrategy.name ^ "_price_series.csv");
    Visualization.plot_order_book_heatmap (List.hd final_books) (strategy.TradingStrategy.name ^ "_order_book_heatmap.csv");
    Visualization.plot_portfolio_value (List.mapi (fun i _ ->
      portfolio.cash +. List.fold_left (fun acc (asset_id, qty) ->
        let price = List.nth stats.mid_prices i |> snd in
        acc +. float_of_int qty *. price
      ) 0. portfolio.positions
    ) stats.mid_prices) (strategy.TradingStrategy.name ^ "_portfolio_value.csv");
  ) strategies final_portfolios;

  printf "Total market orders: %d\n" stats.num_market_orders;
  printf "Total limit orders: %d\n" stats.num_limit_orders;
  printf "Total cancellations: %d\n" stats.num_cancellations;
  printf "Total price changes: %d\n" stats.price_changes;
  printf "Total volume traded: %d\n" stats.volume_traded