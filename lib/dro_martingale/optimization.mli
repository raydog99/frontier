open Torch

val gradient_descent : (Tensor.t -> Tensor.t) -> Tensor.t -> float -> int -> Tensor.t
val optimal_hedge : (Tensor.t -> Tensor.t -> Tensor.t) -> Tensor.t -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> Tensor.t
val optimal_semi_static_hedge : (Tensor.t -> Tensor.t -> Tensor.t) -> Tensor.t -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> Tensor.t