open Torch

type method_t = 
  | EqualWeight
  | MeanVariance of float
  | RiskParity
  | BlackLitterman of (Tensor.t * Tensor.t)

type t

val create : method_t -> int -> t
val construct_portfolio : t -> Tensor.t -> Tensor.t -> Tensor.t