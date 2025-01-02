open Torch
open Elicitable_functional

type t = {
  functional : Elicitable_functional.t;
  coefficients : Tensor.t;
}

val create : Elicitable_functional.t -> int -> t
val predict : t -> Tensor.t -> Tensor.t
val loss : t -> Tensor.t -> Tensor.t -> Tensor.t
val fit : t -> Tensor.t -> Tensor.t -> float -> int -> unit