open Order_book_model

type t = {
  bids: (int * int) list;
  asks: (int * int) list;
  asset_id: int;
  hidden_orders: (side * int * int) list;
}

val empty : int -> t
val best_bid : t -> int option
val best_ask : t -> int option
val add_order : t -> order -> t
val remove_order : t -> order -> t
val match_orders : t -> t
val get_mid_price : t -> int
val get_spread : t -> int
val get_volume_at_distance : int -> t -> side -> int
val get_book_depth : t -> int
val get_book_imbalance : t -> float
val get_order_book_snapshot : t -> int array
val apply_market_impact : t -> order -> market_impact -> t * float
val add_iceberg_order : t -> order -> t
val match_hidden_orders : t -> t
val update_book : t -> order -> t