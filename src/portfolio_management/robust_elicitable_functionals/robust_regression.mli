open Torch
open Regression

type t = {
  base_regression : Regression.t;
  epsilon : float;
}

val create : Regression.t -> float -> t
val optimize_eta : t -> Tensor.t -> Tensor.t -> float
val loss : t -> Tensor.t -> Tensor.t -> Tensor.t
val fit : t -> Tensor.t -> Tensor.t -> float -> int -> unit