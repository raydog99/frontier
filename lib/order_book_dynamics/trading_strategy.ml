open Order_book_model

type t = {
  name: string;
  decide: order_book list -> portfolio -> order option;
}

let market_making ~spread_threshold ~inventory_limit =
  let name = "Market Making" in
  let decide books portfolio =
    let book = List.hd books in
    let mid_price = OrderBook.get_mid_price book in
    let spread = OrderBook.get_spread book in
    let inventory = List.assoc_opt book.asset_id portfolio.positions |> Option.value ~default:0 in
    if spread > spread_threshold then
      if inventory < inventory_limit then
        Some { order_type = Limit; side = Buy; price = mid_price - spread_threshold / 2; size = 1; asset_id = book.asset_id; hidden_size = None }
      else if inventory > -inventory_limit then
        Some { order_type = Limit; side = Sell; price = mid_price + spread_threshold / 2; size = 1; asset_id = book.asset_id; hidden_size = None }
      else None
    else None
  in
  {name; decide}

let momentum ~lookback ~threshold =
  let name = "Momentum" in
  let price_history = ref [] in
  let decide books portfolio =
    let book = List.hd books in
    let mid_price = float_of_int (OrderBook.get_mid_price book) in
    price_history := (mid_price :: !price_history) |> List.take lookback;
    if List.length !price_history = lookback then
      let returns = List.map2 (fun p1 p2 -> (p2 -. p1) /. p1) (List.tl !price_history) !price_history in
      let avg_return = List.fold_left (+.) 0. returns /. float_of_int (List.length returns) in
      if avg_return > threshold then
        Some { order_type = Market; side = Buy; price = 0; size = 1; asset_id = book.asset_id; hidden_size = None }
      else if avg_return < -.threshold then
        Some { order_type = Market; side = Sell; price = 0; size = 1; asset_id = book.asset_id; hidden_size = None }
      else None
    else None
  in
  {name; decide}

let simple_neural_network ~input_size ~hidden_size ~output_size =
  let name = "Simple Neural Network" in
  let w1 = Array.make_matrix input_size hidden_size 0. in
  let w2 = Array.make_matrix hidden_size output_size 0. in
  let b1 = Array.make hidden_size 0. in
  let b2 = Array.make output_size 0. in

  (* Initialize weights with random values *)
  for i = 0 to input_size - 1 do
    for j = 0 to hidden_size - 1 do
      w1.(i).(j) <- Random.float 2. -. 1.
    done
  done;
  for i = 0 to hidden_size - 1 do
    for j = 0 to output_size - 1 do
      w2.(i).(j) <- Random.float 2. -. 1.
    done
  done;

  let relu x = max 0. x in
  let sigmoid x = 1. /. (1. +. exp (-. x)) in

  let forward input =
    let hidden = Array.make hidden_size 0. in
    for i = 0 to hidden_size - 1 do
      for j = 0 to input_size - 1 do
        hidden.(i) <- hidden.(i) +. input.(j) *. w1.(j).(i)
      done;
      hidden.(i) <- relu (hidden.(i) +. b1.(i))
    done;
    let output = Array.make output_size 0. in
    for i = 0 to output_size - 1 do
      for j = 0 to hidden_size - 1 do
        output.(i) <- output.(i) +. hidden.(j) *. w2.(j).(i)
      done;
      output.(i) <- sigmoid (output.(i) +. b2.(i))
    done;
    output
  in

  let decide books portfolio =
    let book = List.hd books in
    let input = [|
      float_of_int (OrderBook.get_mid_price book);
      float_of_int (OrderBook.get_spread book);
      OrderBook.get_book_imbalance book;
      float_of_int (OrderBook.get_book_depth book);
    |] in
    let output = forward input in
    if output.(0) > 0.5 then
      Some { order_type = Market; side = Buy; price = 0; size = 1; asset_id = book.asset_id; hidden_size = None }
    else if output.(0) < 0.5 then
      Some { order_type = Market; side = Sell; price = 0; size = 1; asset_id = book.asset_id; hidden_size = None }
    else
      None
  in
  {name; decide}

let pairs_trading ~correlation_threshold ~zscore_threshold =
  let name = "Pairs Trading" in
  let price_history = Hashtbl.create 10 in
  let correlation_matrix = Hashtbl.create 10 in

  let update_price_history asset_id price =
    let history = Hashtbl.find_opt price_history asset_id |> Option.value ~default:[] in
    let new_history = (price :: history) |> List.take 100 in
    Hashtbl.replace price_history asset_id new_history
  in

  let calculate_correlation asset_id1 asset_id2 =
    let history1 = Hashtbl.find price_history asset_id1 in
    let history2 = Hashtbl.find price_history asset_id2 in
    let (slope, intercept) = Analysis.linear_regression history1 history2 in
    let residuals = List.map2 (fun p1 p2 -> p2 -. (slope *. p1 +. intercept)) history1 history2 in
    let mean = List.fold_left (+.) 0. residuals /. float_of_int (List.length residuals) in
    let std_dev = sqrt (List.fold_left (fun acc r -> acc +. (r -. mean) ** 2.) 0. residuals /. float_of_int (List.length residuals)) in
    (slope, intercept, mean, std_dev)
  in

  let decide books portfolio =
    let book = List.hd books in
    let price = float_of_int (OrderBook.get_mid_price book) in
    update_price_history book.asset_id price;

    if Hashtbl.length price_history >= 2 then
      let asset_ids = Hashtbl.to_seq_keys price_history |> List.of_seq in
      List.iter (fun id1 ->
        List.iter (fun id2 ->
          if id1 < id2 then
            let key = (min id1 id2, max id1 id2) in
            let (slope, intercept, mean, std_dev) = calculate_correlation id1 id2 in
            Hashtbl.replace correlation_matrix key (slope, intercept, mean, std_dev)
        ) asset_ids
      ) asset_ids;

      let current_asset = book.asset_id in
      let paired_asset_opt = Hashtbl.to_seq correlation_matrix
        |> Seq.find_map (fun ((id1, id2), (slope, intercept, mean, std_dev)) ->
            if id1 = current_asset || id2 = current_asset then
              let other_asset = if id1 = current_asset then id2 else id1 in
              let correlation = abs slope in
              if correlation > correlation_threshold then
                Some (other_asset, slope, intercept, mean, std_dev)
              else None
            else None)
      in

      match paired_asset_opt with
      | Some (paired_asset, slope, intercept, mean, std_dev) ->
          let current_price = Hashtbl.find price_history current_asset |> List.hd in
          let paired_price = Hashtbl.find price_history paired_asset |> List.hd in
          let expected_price = slope *. current_price +. intercept in
          let zscore = (paired_price -. expected_price) /. std_dev in
          if abs zscore > zscore_threshold then
            if zscore > 0. then
              Some { order_type = Market; side = Sell; price = 0; size = 1; asset_id = paired_asset; hidden_size = None }
            else
              Some { order_type = Market; side = Buy; price = 0; size = 1; asset_id = paired_asset; hidden_size = None }
          else None
      | None -> None
    else None
  in
  {name; decide}