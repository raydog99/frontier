type t = {
  mutable price: float;
  mutable quantity: float;
}

let create initial_price initial_quantity =
  { price = initial_price; quantity = initial_quantity }

let clear market demand supply =
  let price_elasticity = 0.1 in
  let price_change = price_elasticity *. (demand -. supply) /. supply in
  market.price <- market.price *. (1. +. price_change);
  market.quantity <- min demand supply