open Torch

type adaptive_params = {
  scale: Tensor.t;
  translation: Tensor.t;
  frequency: Tensor.t option;
  shape: Tensor.t option;
}

type strategy =
  | Gradient
  | MetaLearning
  | Evolutionary

type t

val create : int -> int -> strategy -> t
val forward : t -> Tensor.t -> Tensor.t
val adapt : t -> t