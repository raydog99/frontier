open Torch
open Types
open Config

val indicator_function : Tensor.t -> Tensor.t
val h : t -> int -> float
val saturation_factor : framework -> int -> float
val step_size : t -> int -> float
val optimal_iterations : t -> float -> framework -> int -> int