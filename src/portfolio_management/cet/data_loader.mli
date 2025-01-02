open Torch
open Dataset

type t
val create : Dataset.t -> int -> bool -> t
val iter : t -> f:(Cet.price_volume_data * Cet.earnings_data * Tensor.t -> unit) -> unit