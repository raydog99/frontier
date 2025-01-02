type t = {
  mutable price: float;
  mutable volume: int;
}

let create initial_price =
  { price = initial_price; volume = 0 }

let update_price market demand supply =
  let price_elasticity = 0.1 in
  let price_change = price_elasticity *. (demand -. supply) /. supply in
  market.price <- market.price *. (1. +. price_change);
  market.volume <- int_of_float (min demand supply)