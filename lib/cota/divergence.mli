open Torch

type t = 
  | KL
  | Jensen_Shannon
  | Custom of (Tensor.t -> Tensor.t -> Tensor.t)

val compute : t -> Tensor.t -> Tensor.t -> Tensor.t