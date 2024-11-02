open Torch

val entropy : Tensor.t -> Tensor.t
val cost_matrix : 
  Tensor.t -> Tensor.t -> Types.intervention list -> 
  (Types.intervention -> Types.intervention) -> Tensor.t
val sinkhorn : 
  Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t
val gradient_step : Tensor.t -> Tensor.t -> float -> Tensor.t