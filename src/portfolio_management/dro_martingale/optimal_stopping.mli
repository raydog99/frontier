open Torch

val value_function : (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> Tensor.t
val optimal_stopping_time : (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> Tensor.t
val sensitivity : Tensor.t -> (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> Tensor.t
val adapted_sensitivity : Tensor.t -> (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> Tensor.t
val optimal_stopping_sensitivity_formula : Tensor.t -> (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> Tensor.t