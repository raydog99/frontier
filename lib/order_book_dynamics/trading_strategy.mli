open Order_book_model

type t = {
  name: string;
  decide: order_book list -> portfolio -> order option;
}

val market_making : spread_threshold:int -> inventory_limit:int -> t
val momentum : lookback:int -> threshold:float -> t
val simple_neural_network : input_size:int -> hidden_size:int -> output_size:int -> t
val pairs_trading : correlation_threshold:float -> zscore_threshold:float -> t