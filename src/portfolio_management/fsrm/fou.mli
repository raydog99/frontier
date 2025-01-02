open Torch

type t

val create : hurst:float -> eta:float -> kappa:float -> t
val sample : t -> int -> Tensor.t
val autocorrelation : t -> float -> float
val variance : t -> float
val conditional_probability : t -> float -> float -> float
val minimum_autocorrelation : t -> float * float
val optimal_lag : t -> float