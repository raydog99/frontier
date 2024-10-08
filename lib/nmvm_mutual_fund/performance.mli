type t = {
  cumulative_return: float;
  annualized_return: float;
  annualized_volatility: float;
  sharpe_ratio: float;
  sortino_ratio: float;
  maximum_drawdown: float;
  value_at_risk: float;
  conditional_value_at_risk: float;
  calmar_ratio: float;
  omega_ratio: float;
  information_ratio: float;
  treynor_ratio: float;
}

val calculate : Nmvm.t -> Portfolio.t -> float -> float -> int -> Portfolio.t option -> t
val to_string : t -> string