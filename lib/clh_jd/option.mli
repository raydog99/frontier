open Torch

type option_type = Call | Put

type t

val create : option_type:option_type -> strike:float -> maturity:float -> t

val black_scholes_price : 
  t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

val black_scholes_delta :
  t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t