open Torch

val rolling_window : (Tensor.t -> float) -> Tensor.t -> int -> Tensor.t
val moving_average : Tensor.t -> int -> Tensor.t
val exponential_moving_average : Tensor.t -> float -> Tensor.t
val scale_data : Tensor.t -> Tensor.t
val train_test_split : Tensor.t -> float -> Tensor.t * Tensor.t