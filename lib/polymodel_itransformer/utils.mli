open Torch

val load_data : string -> Tensor.t
val preprocess_data : Tensor.t -> Tensor.t
val evaluate_performance : Tensor.t -> Tensor.t -> float
val split_data : Tensor.t -> float -> Tensor.t * Tensor.t
val create_sequences : Tensor.t -> int -> Tensor.t
val create_dataset : Tensor.t -> int -> Dataset.t