type performance = {
  return: float;
  risk: float;
  sharpe_ratio: float;
  max_drawdown: float;
}

val calculate_return : float list -> float
val calculate_risk : float list -> float
val calculate_sharpe_ratio : float -> float -> float -> float
val calculate_max_drawdown : float list -> float
val evaluate_performance : float list -> float -> performance