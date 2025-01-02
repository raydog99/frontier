open Types
open Torch

val evolve_order_parameter : order_parameter -> area -> float -> order_parameter
val compute_returns : Tensor.t -> int -> Tensor.t
val compute_moment : Tensor.t -> int -> float
val compute_hurst_exponent : Tensor.t -> int -> float
val equilibrium_return_distribution : Tensor.t -> return_distribution
val generalized_hyperbolic_distribution : Tensor.t -> genus -> return_distribution
val analyze_multifractal_scaling : Tensor.t -> int -> int -> multifractal_spectrum