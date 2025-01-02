open Torch

type t

val create : float -> float -> int -> float -> float -> float -> float -> float -> t
val optimize : t -> Tensor.t -> Tensor.t
val objective_function : t -> Tensor.t -> Tensor.t
val optimize_almgren_chriss : t -> Tensor.t -> Tensor.t
val optimize_dynamic : t -> Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t
val optimize_constrained : t -> Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t