type option_type = Call | Put

type option = {
  underlying: string;
  option_type: option_type;
  strike: float;
  expiry: float;
  price: float;
}

val black_scholes : float -> float -> float -> float -> float -> option_type -> float
val calculate_delta : float -> float -> float -> float -> float -> option_type -> float
val calculate_gamma : float -> float -> float -> float -> float -> float
val calculate_theta : float -> float -> float -> float -> float -> option_type -> float
val delta_hedge_strategy : float -> option -> float -> int
val monte_carlo_option_pricing : float -> float -> float -> float -> float -> option_type -> int -> float