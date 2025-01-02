type market = private {
  stock: Tree.t;
  derivative: Perpetual_derivative.t;
  option: Option.t;
  risk_free_rate: float;
}

type scenario = {
  delta_s: float;
  delta_sigma: float;
  delta_r: float;
  delta_q: float;
  jump_intensity: float;
  jump_size: float;
}

val create_market : Tree.t -> Perpetual_derivative.t -> Option.t -> float -> market
val is_complete : market -> bool
val price_option : market -> float
val replicating_portfolio : market -> float -> float -> float * float * float
val hedge_error : market -> float -> float -> float -> float
val simulate_hedge_performance : market -> float -> float -> int -> int -> float list list
val value_at_risk : market -> float -> int -> float -> float
val expected_shortfall : market -> float -> int -> float -> float
val incremental_var : market -> (float * float) list -> float -> int -> float -> (float * float * float) list
val component_var : market -> (float * float) list -> float -> int -> float -> (float * float * float) list
val marginal_var : market -> (float * float) list -> float -> int -> float -> (float * float) list
val calc_greeks : market -> { delta: float; gamma: float; theta: float; vega: float; rho: float }
val apply_scenario : market -> scenario -> market
val stress_test : market -> scenario list -> (scenario * float * { delta: float; gamma: float; theta: float; vega: float; rho: float }) list