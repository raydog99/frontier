open Torch
open Elicitable_functional

type t = {
  functional: Elicitable_functional.t;
  b_values: float array;
  distribution: Tensor.t;
}

val create : Elicitable_functional.t -> float array -> Tensor.t -> t
val evaluate : t -> float array