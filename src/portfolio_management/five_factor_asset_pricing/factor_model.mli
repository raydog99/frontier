open Torch

type t

val create : float -> t
val train : t -> Tensor.t -> Tensor.t -> int -> int -> t
val predict : t -> Tensor.t -> Tensor.t
val calculate_r_squared : t -> Tensor.t -> Tensor.t -> float
val get_weights : t -> float array