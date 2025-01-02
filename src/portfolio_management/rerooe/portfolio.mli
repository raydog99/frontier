type t

type asset = {
  model: Model.t;
  weight: float ref;
  sector: string;
}

type performance_summary = {
  total_return: float;
  sharpe_ratio: float;
  max_drawdown: float;
}

val create : Model.t array -> float array -> t
val rebalance : t -> float array -> unit
val update_asset_price : t -> string -> float -> float -> unit
val execute_order : t -> string -> int -> float -> unit
val get_returns : t -> float list
val get_weights : t -> float array
val get_assets : t -> asset array
val get_performance_summary : t -> performance_summary
val get_total_value : t -> float
val get_asset_values : t -> float array