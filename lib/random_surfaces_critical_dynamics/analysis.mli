open Types
open Torch

val analyze_regime_transitions : float list -> int list -> (regime * float * int) list
val compute_autocorrelation : Tensor.t -> int -> Tensor.t
val estimate_lyapunov_exponent : float list -> float
val compute_fractal_dimension : Tensor.t -> int -> float