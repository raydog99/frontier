open Torch
open Types

type t

val create : kan_params -> t
val forward : t -> Tensor.t -> Tensor.t
val parameters : t -> Tensor.t list