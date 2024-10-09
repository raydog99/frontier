type t = private {
  tree: Tree.t;
  gamma: float;
}

val create : Tree.t -> t
val price : t -> float -> float
val drift : t -> float
val volatility : t -> float
val delta : t -> float -> float
val gamma : t -> float -> float
val theta : t -> float -> float
val vega : t -> float -> float
val rho : t -> float -> float