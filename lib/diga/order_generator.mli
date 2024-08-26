open Torch

type order = {
	timestamp: float;
	price: float;
	volume: int;
	is_buy: bool;
}

type t

val create : initial_price:float -> risk_aversion:float -> fundamental_weight:float -> chartist_weight:float -> noise_weight:float -> t
val generate_orders : t -> Tensor.t -> order list