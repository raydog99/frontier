type option_type = Call | Put
type style = European | American | Bermudan of float list
type barrier_type = UpAndOut | UpAndIn | DownAndOut | DownAndIn
type exotic_type =
  | Vanilla
  | Barrier of barrier_type * float
  | AsianFixed of int
  | Lookback

type t = private {
  tree: Tree.t;
  option_type: option_type;
  strike: float;
  style: style;
  exotic: exotic_type;
}

type multi_asset_option = private {
  trees: Tree.multi_asset_t;
  option_type: option_type;
  strike: float;
  style: style;
  exotic: exotic_type;
}

val create : Tree.t -> option_type -> float -> style -> exotic_type -> t
val create_multi_asset : Tree.multi_asset_t -> option_type -> float -> style -> exotic_type -> multi_asset_option
val payoff : t -> float -> float
val price : t -> float
val price_with_control_variate : t -> float
val price_with_importance_sampling : t -> int -> float
val delta : t -> float -> float -> float
val gamma : t -> float -> float -> float
val theta : t -> float -> float -> float
val vega : t -> float -> float -> float
val rho : t -> float -> float -> float
val finite_difference_greeks : t -> { delta: float; gamma: float; theta: float; vega: float; rho: float }