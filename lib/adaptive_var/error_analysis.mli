open Torch

type error_stats = {
  mse: float;
  mae: float;
  max_error: float;
  bias: float;
  std_dev: float;
}

val compute_error_stats : Tensor.t -> Tensor.t -> error_stats
val confidence_interval : float -> error_stats -> float -> float * float
val relative_error : float -> float -> float