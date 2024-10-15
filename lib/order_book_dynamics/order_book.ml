type t = {
  bids: (int * int) list;
  asks: (int * int) list;
  asset_id: int;
  hidden_orders: (Order_book_model.side * int * int) list;
}

let empty asset_id = { bids = []; asks = []; asset_id; hidden_orders = [] }

let best_bid book = List.hd_opt book.bids |> Option.map fst
let best_ask book = List.hd_opt book.asks |> Option.map fst

let add_order book order =
  match order.Order_book_model.order_type with
  | Market -> book  (* Market orders are executed immediately, not added to the book *)
  | Cancel -> book  (* Cancel orders are handled separately *)
  | Limit | Iceberg ->
      if order.side = Order_book_model.Buy then
        { book with bids = (order.price, order.size) :: book.bids |> List.sort (fun (a,_) (b,_) -> compare b a) }
      else
        { book with asks = (order.price, order.size) :: book.asks |> List.sort compare }

let remove_order book order =
  let remove_from_list price size = List.filter (fun (p, s) -> p != price || s != size) in
  match order.Order_book_model.order_type with
  | Cancel ->
      if order.side = Order_book_model.Buy then
        { book with bids = remove_from_list order.price order.size book.bids }
      else
        { book with asks = remove_from_list order.price order.size book.asks }
  | _ -> book  (* Only cancel orders remove from the book *)

let match_orders book =
  match book.bids, book.asks with
  | (bid_price, bid_size) :: rem_bids, (ask_price, ask_size) :: rem_asks when bid_price >= ask_price ->
      let matched_size = min bid_size ask_size in
      let new_bids = if bid_size > matched_size then (bid_price, bid_size - matched_size) :: rem_bids else rem_bids in
      let new_asks = if ask_size > matched_size then (ask_price, ask_size - matched_size) :: rem_asks else rem_asks in
      { book with bids = new_bids; asks = new_asks }
  | _ -> book

let get_mid_price book =
  match best_bid book, best_ask book with
  | Some bid, Some ask -> (bid + ask) / 2
  | _ -> 0  (* Default to 0 if book is empty *)

let get_spread book =
  match best_bid book, best_ask book with
  | Some bid, Some ask -> ask - bid
  | _ -> 0  (* Default to 0 if book is empty *)

let get_volume_at_distance i book side =
  let prices = if side = Order_book_model.Buy then book.bids else book.asks in
  let best_price = if side = Order_book_model.Buy then best_bid book else best_ask book in
  match best_price with
  | Some p ->
      let target_price = if side = Order_book_model.Buy then p - i else p + i in
      List.find_opt (fun (price, _) -> price = target_price) prices
      |> Option.map snd
      |> Option.value ~default:0
  | None -> 0

let get_book_depth book =
  max (List.length book.bids) (List.length book.asks)

let get_book_imbalance book =
  let total_bid_volume = List.fold_left (fun acc (_, size) -> acc + size) 0 book.bids in
  let total_ask_volume = List.fold_left (fun acc (_, size) -> acc + size) 0 book.asks in
  float_of_int (total_bid_volume - total_ask_volume) /. float_of_int (total_bid_volume + total_ask_volume)

let get_order_book_snapshot book =
  let max_depth = 5 in
  let snapshot = Array.make (2 * max_depth) 0 in
  List.iteri (fun i (_, size) -> if i < max_depth then snapshot.(i) <- size) book.bids;
  List.iteri (fun i (_, size) -> if i < max_depth then snapshot.(max_depth + i) <- size) book.asks;
  snapshot

let apply_market_impact book order impact =
  let mid_price = float_of_int (get_mid_price book) in
  let temp_impact = impact.Order_book_model.temporary mid_price order.Order_book_model.size in
  let perm_impact = impact.Order_book_model.permanent mid_price order.Order_book_model.size in
  let new_mid_price = int_of_float (mid_price +. perm_impact) in
  let new_bids = List.map (fun (p, s) -> (p + new_mid_price - get_mid_price book, s)) book.bids in
  let new_asks = List.map (fun (p, s) -> (p + new_mid_price - get_mid_price book, s)) book.asks in
  {book with bids = new_bids; asks = new_asks}, temp_impact

let add_iceberg_order book order =
  let visible_size = min order.Order_book_model.size (Option.value order.Order_book_model.hidden_size ~default:order.Order_book_model.size) in
  let hidden_size = order.Order_book_model.size - visible_size in
  let updated_book = add_order book {order with size = visible_size} in
  let hidden_orders = 
    if hidden_size > 0 then
      (order.Order_book_model.side, order.Order_book_model.price, hidden_size) :: book.hidden_orders
    else
      book.hidden_orders
  in
  {updated_book with hidden_orders = hidden_orders}

let match_hidden_orders book =
  let (matched_orders, remaining_hidden) = List.partition (fun (side, price, _) ->
    match side with
    | Order_book_model.Buy -> price >= (best_ask book |> Option.value ~default:max_int)
    | Order_book_model.Sell -> price <= (best_bid book |> Option.value ~default:0)
  ) book.hidden_orders in
  
  let updated_book = List.fold_left (fun b (side, price, size) ->
    add_order b {Order_book_model.order_type = Limit; side; price; size; asset_id = book.asset_id; hidden_size = None}
  ) book matched_orders in
  
  {updated_book with hidden_orders = remaining_hidden}

let update_book book order =
  match order.Order_book_model.order_type with
  | Iceberg -> add_iceberg_order book order |> match_hidden_orders
  | _ -> add_order book order |> match_orders |> match_hidden_orders