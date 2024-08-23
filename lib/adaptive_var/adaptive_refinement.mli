open Torch
open Types
open Config

val eta_n_l : t -> framework -> float -> int -> int -> int -> float
val refine : t -> framework -> Types.loss_function -> Tensor.t -> Tensor.t -> float -> int -> int -> Tensor.t