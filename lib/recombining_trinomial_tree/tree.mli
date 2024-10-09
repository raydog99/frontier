type t = private {
  s0: float;
  r: float;
  sigma: float;
  t: float;
  n: int;
  u: float;
  d: float;
  pu: float;
  pm: float;
  pd: float;
  dt: float;
}

type multi_asset_t = private {
  assets: t array;
  correlation_matrix: float array array;
}

val create : float -> float -> float -> float -> int -> t
val create_with_parameters : float -> float -> float -> float -> int -> float -> float -> float -> float -> float -> t
val create_multi_asset : t array -> float array array -> multi_asset_t
val stock_price : t -> int -> int -> float
val discount_factor : t -> float
val risk_neutral_probabilities : t -> float * float * float
val update_parameters : t -> float -> float -> float -> float -> float -> t
val expected_return : t -> float
val variance : t -> float
val generate_correlated_returns : multi_asset_t -> float array
val create_from_stochastic_process : Stochastic_process.t -> float -> float -> float -> int -> t
val optimize_parameters : float array -> t -> t