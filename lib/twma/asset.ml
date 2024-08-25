open Black_scholes

type asset_type =
  | Stock
  | Option of { underlying: string; strike: float; expiry: float; is_call: bool }
  | Future of { underlying: string; expiry: float }
  | Forex of { base_currency: string; quote_currency: string }

type t = {
  symbol: string;
  asset_type: asset_type;
  mutable price: float;
  mutable fundamental_data: (string * float) list;
}

let create symbol asset_type =
  { symbol; asset_type; price = 0.; fundamental_data = [] }

let update_price t new_price =
  t.price <- new_price

let update_fundamental t key value =
  t.fundamental_data <- (key, value) :: List.remove_assoc key t.fundamental_data

let get_fundamental t key =
  List.assoc_opt key t.fundamental_data

let calculate_implied_volatility t market_price =
  match t.asset_type with
  | Option { underlying; strike; expiry; is_call } ->
      let time_to_expiry = expiry -. (float_of_int (int_of_float (Unix.time ()))) /. (365. *. 24. *. 60. *. 60.) in
      let underlying_price = 0. in
      let rf_rate = 0.02 in
      Black_scholes.implied_volatility market_price underlying_price strike time_to_expiry rf_rate is_call
  | _ -> failwith "Implied volatility calculation is only applicable to options"

let calculate_basis t =
  match t.asset_type with
  | Future { underlying; _ } ->
      let spot_price = 0. in
      t.price -. spot_price
  | _ -> failwith "Basis calculation is only applicable to futures"