open Torch

val rmvnorm : Tensor.t -> Tensor.t -> Tensor.t
val rgamma : Tensor.t -> Tensor.t -> Tensor.t
val linear_predictor : Tensor.t -> Tensor.t -> 
                      Tensor.t -> Tensor.t -> Tensor.t