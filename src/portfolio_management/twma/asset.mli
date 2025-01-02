type asset_type =
  | Stock
  | Option of { underlying: string; strike: float; expiry: float; is_call: bool }
  | Future of { underlying: string; expiry: float }
  | Forex of { base_currency: string; quote_currency: string }

type t

val create : string -> asset_type -> t
val update_price : t -> float -> unit
val update_fundamental : t -> string -> float -> unit
val get_fundamental : t -> string -> float option
val calculate_implied_volatility : t -> float -> float
val calculate_basis : t -> float
