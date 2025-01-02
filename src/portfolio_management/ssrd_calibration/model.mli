open Torch
open Types

val psi : float -> float -> float -> float
val theta : float -> float -> float -> float -> float
val v0 : params -> float -> float -> float -> float -> float
val h0 : params -> float -> float -> float -> float -> float
val v : params -> float -> float -> float -> float -> approximation_order -> float
val h : params -> float -> float -> float -> float -> approximation_order -> float
val simulate_ssrd : params -> float -> Tensor.t * Tensor.t