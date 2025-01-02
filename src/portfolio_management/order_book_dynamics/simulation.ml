open Order_book_model

let k = 1.92  (* Power law parameter *)
let alpha = 0.52  (* Power law parameter *)
let mu = 0.94  (* Market order arrival rate *)
let theta i = 0.71 *. exp (-0.0769 *. float_of_int i)  (* Cancellation rate function *)

let lambda i = k /. (float_of_int i ** alpha)  (* Limit order arrival rate function *)

let generate_limit_order book side =
  let rec place_order i =
    let rate = lambda i in
    if Random.float 1.0 < rate then
      match side with
      | Buy -> 
          let price = Option.value (OrderBook.best_ask book) ~default:100 - i in
          { order_type = Limit; side = Buy; price; size = 1; asset_id = book.asset_id; hidden_size = None }
      | Sell -> 
          let price = Option.value (OrderBook.best_bid book) ~default:0 + i in
          { order_type = Limit; side = Sell; price; size = 1; asset_id = book.asset_id; hidden_size = None }
    else
      place_order (i + 1)
  in
  place_order 1

let generate_order book =
  let total_rate = mu +. (lambda 1) *. 2.0 +. 
    (List.length book.bids + List.length book.asks |> float_of_int) *. theta 1 in
  let rand = Random.float total_rate in
  if rand < mu then
    { order_type = Market; side = if Random.bool () then Buy else Sell; price = 0; size = 1; asset_id = book.asset_id; hidden_size = None }
  else if rand < mu +. (lambda 1) *. 2.0 then
    generate_limit_order book (if Random.bool () then Buy else Sell)
  else
    let cancel_from_side side =
      match side with
      | Buy -> 
          if List.length book.bids > 0 then
            let idx = Random.int (List.length book.bids) in
            let (price, _) = List.nth book.bids idx in
            { order_type = Cancel; side = Buy; price; size = 1; asset_id = book.asset_id; hidden_size = None }
          else
            generate_order book  (* Retry if no bids to cancel *)
      | Sell ->
          if List.length book.asks > 0 then
            let idx = Random.int (List.length book.asks) in
            let (price, _) = List.nth book.asks idx in
            { order_type = Cancel; side = Sell; price; size = 1; asset_id = book.asset_id; hidden_size = None }
          else
            generate_order book  (* Retry if no asks to cancel *)
    in
    cancel_from_side (if Random.bool () then Buy else Sell)

let create_stats () = {
  num_market_orders = 0;
  num_limit_orders = 0;
  num_cancellations = 0;
  price_changes = 0;
  volume_traded = 0;
  mid_prices = [];
}

let update_stats stats order book new_book =
  match order.order_type with
  | Market -> 
      stats.num_market_orders <- stats.num_market_orders + 1;
      stats.volume_traded <- stats.volume_traded + order.size
  | Limit -> stats.num_limit_orders <- stats.num_limit_orders + 1
  | Cancel -> stats.num_cancellations <- stats.num_cancellations + 1
  | Iceberg -> 
      stats.num_limit_orders <- stats.num_limit_orders + 1;
      stats.volume_traded <- stats.volume_traded + (order.size - Option.value order.hidden_size ~default:0);
  if OrderBook.get_mid_price book <> OrderBook.get_mid_price new_book then
    stats.price_changes <- stats.price_changes + 1;
  stats.mid_prices <- (book.asset_id, float_of_int (OrderBook.get_mid_price new_book)) :: stats.mid_prices

let run_event_driven iterations initial_books strategies =
  let stats = create_stats () in
  let event_queue = Queue.create () in
  let portfolios = List.map (fun _ -> {cash = 1000000.; positions = []}) strategies in

  let process_event event books =
    match event with
    | OrderEvent order ->
        let book = List.find (fun b -> b.asset_id = order.asset_id) books in
        let new_book = OrderBook.update_book book order in
        let new_books = List.map (fun b -> if b.asset_id = order.asset_id then new_book else b) books in
        update_stats stats order book new_book;
        if OrderBook.get_mid_price book <> OrderBook.get_mid_price new_book then
          Queue.add (MarketDataEvent new_book) event_queue;
        new_books
    | TradeEvent (order, price) ->
        let portfolio = List.nth portfolios (order.asset_id mod List.length portfolios) in
        let trade_value = float_of_int (price * order.size) in
        (match order.side with
        | Buy -> 
            portfolio.cash <- portfolio.cash -. trade_value;
            portfolio.positions <- (order.asset_id, (List.assoc_opt order.asset_id portfolio.positions |> Option.value ~default:0) + order.size) :: 
                                   List.remove_assoc order.asset_id portfolio.positions
        | Sell ->
            portfolio.cash <- portfolio.cash +. trade_value;
            portfolio.positions <- (order.asset_id, (List.assoc_opt order.asset_id portfolio.positions |> Option.value ~default:0) - order.size) ::
                                   List.remove_assoc order.asset_id portfolio.positions);
        books
    | MarketDataEvent new_book ->
        List.iter (fun strategy ->
          let order_opt = strategy.TradingStrategy.decide [new_book] (List.nth portfolios (new_book.asset_id mod List.length portfolios)) in
          match order_opt with
          | Some order -> Queue.add (OrderEvent order) event_queue
          | None -> ()
        ) strategies;
        books
  in

  let rec loop n books =
    if n = 0 then (books, stats, portfolios)
    else if Queue.is_empty event_queue then
      let order = generate_order (List.nth books (Random.int (List.length books))) in
      Queue.add (OrderEvent order) event_queue;
      loop n books
    else
      let event = Queue.take event_queue in
      let new_books = process_event event books in
      loop (n-1) new_books
  in

  List.iter (fun book -> Queue.add (MarketDataEvent book) event_queue) initial_books;
  loop iterations initial_books