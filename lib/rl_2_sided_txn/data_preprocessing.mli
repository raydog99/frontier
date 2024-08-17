open Torch

exception Invalid_data_shape of string

val normalize_prices : Tensor.t -> Tensor.t
val preprocess_data : Tensor.t -> Tensor.t
val load_and_preprocess_csv : string -> Tensor.t