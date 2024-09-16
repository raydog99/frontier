open Torch

val calculate_implementation_shortfall : Tensor.t -> Tensor.t -> float -> float
val calculate_vwap : Tensor.t -> Tensor.t -> float
val calculate_participation_rate : Tensor.t -> float -> float array