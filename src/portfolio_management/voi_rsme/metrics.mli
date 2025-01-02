open Torch

val rmse : Tensor.t -> Tensor.t -> float
val correlation : Tensor.t -> Tensor.t -> float
val mean_rate_of_return : Tensor.t -> Tensor.t -> float