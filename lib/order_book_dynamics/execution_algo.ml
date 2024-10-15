open Order_book_model

let twap ~duration ~num_slices =
  let name = "Time-Weighted Average Price" in
  let execute order books remaining_time =
    let slice_size = max 1 (order.size / num_slices) in
    let time_per_slice = duration / num_slices in
    if remaining_time <= 0 then []
    else if remaining_time < time_per_slice then
      [{order with size = order.size}]
    else
      let current_slice = {order with size = slice_size} in
      current_slice :: execute {order with size = order.size - slice_size} books (remaining_time - time_per_slice)
  in
  {name; execute}

let vwap ~duration ~num_slices =
  let name = "Volume-Weighted Average Price" in
  let execute order books remaining_time =
    let total_volume = List.fold_left (fun acc book ->
      acc + List.fold_left (fun vol (_, size) -> vol + size) 0 book.bids +
      List.fold_left (fun vol (_, size) -> vol + size) 0 book.asks
    ) 0 books in
    let volume_per_slice = total_volume / num_slices in
    let rec slice_orders acc_volume acc_orders remaining_size =
      if remaining_size <= 0 then List.rev acc_orders
      else
        let current_volume = List.fold_left (fun acc book ->
          acc + List.fold_left (fun vol (_, size) -> vol + size) 0 book.bids +
          List.fold_left (fun vol (_, size) -> vol + size) 0 book.asks
        ) 0 books in
        let slice_size = int_of_float (float_of_int remaining_size *. (float_of_int current_volume /. float_of_int total_volume)) in
        let adjusted_slice_size = max 1 (min slice_size remaining_size) in
        slice_orders (acc_volume + current_volume) ({order with size = adjusted_slice_size} :: acc_orders) (remaining_size - adjusted_slice_size)
    in
    if remaining_time <= 0 then []
    else slice_orders 0 [] order.size
  in
  {name; execute}