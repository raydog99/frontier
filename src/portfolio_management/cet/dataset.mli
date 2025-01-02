open Torch
open Cet

type t
val create : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> int -> t
val length : t -> int
val get : t -> int -> Cet.price_volume_data * Cet.earnings_data * Tensor.t
val get_negative_sample : t -> int -> Cet.price_volume_data * Cet.earnings_data * Tensor.t