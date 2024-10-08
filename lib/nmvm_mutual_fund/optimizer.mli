open Torch

type strategy =
  | Unconstrained
  | Constrained of Portfolio.constraint_type list
  | MeanVariance of float
  | MaximumSharpe
  | MinimumVariance
  | EqualWeight
  | RiskParity
  | BlackLitterman of (Tensor.t * Tensor.t)

val optimize : Nmvm.t -> strategy -> Utility.t option -> float -> float -> Portfolio.t
val cross_validate : Nmvm.t -> strategy -> Utility.t -> float -> float -> int -> float