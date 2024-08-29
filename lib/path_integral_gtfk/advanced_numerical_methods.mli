open Torch

val runge_kutta4 : 
  (float -> Tensor.t -> Tensor.t) -> 
  Tensor.t -> 
  float -> 
  float -> 
  int -> 
  Tensor.t

val finite_difference_greeks :
  option_pricing_fn:(spot:float -> float) ->
  spot:float ->
  strike:float ->
  risk_free_rate:float ->
  volatility:float ->
  maturity:float ->
  option_type:[`Call | `Put] ->
  float * float * float * float

val least_squares_monte_carlo :
  Tensor.t ->
  (Tensor.t -> Tensor.t) ->
  Tensor.t ->
  Tensor.t

val nelder_mead :
  (float array -> float) ->
  float array ->
  float array