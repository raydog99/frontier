open Torch

type t

val create : float -> float -> int -> t
val get_trading_trajectory : t -> Tensor.t
val get_trade_list : t -> Tensor.t