open Torch

type t

val create : hurst:float -> scale:float -> t
val sample : t -> int -> Tensor.t
val increment_variance : t -> float -> float