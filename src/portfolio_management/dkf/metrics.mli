open Torch

val mse : Tensor.t -> Tensor.t -> Tensor.t

val rmse : Tensor.t -> Tensor.t -> Tensor.t

val mae : Tensor.t -> Tensor.t -> Tensor.t

val compute_metrics : Tensor.t -> Tensor.t -> float * float * float