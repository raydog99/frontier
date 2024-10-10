open Torch

val g_derivative : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val distributionally_robust_g : Tensor.t -> Tensor.t -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> Tensor.t
val sensitivity : Tensor.t -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> Tensor.t
val adapted_sensitivity : Tensor.t -> 
  [`Martingale | `Marginal of Tensor.t | `Both of Tensor.t * unit] -> Tensor.t
val forward_start_sensitivity : Tensor.t -> Tensor.t
val forward_start_adapted_sensitivity : Tensor.t -> Tensor.t
val martingale_sensitivity_formula : Tensor.t -> Tensor.t
val adapted_martingale_sensitivity_formula : Tensor.t -> Tensor.t