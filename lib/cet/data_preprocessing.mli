open Torch

val standardize : Tensor.t -> Tensor.t
val dwt : Tensor.t -> float -> Tensor.t
val preprocess_price_volume : Tensor.t -> Tensor.t -> Cet.price_volume_data
val preprocess_earnings : Tensor.t -> Tensor.t
val create_sequences : Tensor.t -> int -> Tensor.t