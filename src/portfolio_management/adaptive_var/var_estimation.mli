open Torch

type method_t =
  | HistoricalSimulation
  | VarianceCovariance
  | MonteCarlo

val estimate_var : method_t -> Tensor.t -> float -> Tensor.t