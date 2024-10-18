open Torch

val estimate_covariance :
  Tensor.t ->
  [ `Tracy_Widom | `Linear_Shrinkage | `Naive ] ->
  Tensor.t