open Torch

type t = {
  generator: Tensor.t;
  rates: Tensor.t;
  state_space: int;
}

val create : Tensor.t -> Tensor.t -> t
val matrix_exp : Tensor.t -> float -> Tensor.t
val g_minus_r : t -> Tensor.t
val jump_intensity : t -> int -> Tensor.t
val jump_probabilities : t -> int -> Tensor.t