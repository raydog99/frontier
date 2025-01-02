type optimization_method =
  | MeanVariance
  | MinimumVariance
  | MaximumSharpeRatio
  | RiskParity

val optimize : Portfolio.t -> optimization_method -> float array