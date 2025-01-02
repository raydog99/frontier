open Torch
open Elicitable_functional

type t = {
  base_functional : Elicitable_functional.t;
  epsilon : float;
}

val create : Elicitable_functional.t -> float -> t
val evaluate : t -> Tensor.t -> Tensor.t