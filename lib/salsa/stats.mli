open Torch

val compute_acf : Tensor.t -> int -> Tensor.t
val ljung_box_test : Tensor.t -> int -> float
val durbin_watson : Tensor.t -> float