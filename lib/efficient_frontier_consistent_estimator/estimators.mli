open Torch
open Efficient_frontier

type t =
  | Sample
  | Consistent
  | SSE
  | EBE
  | RTE

val all : (string * t) list
val estimate : Efficient_frontier.t -> t -> Tensor.t * Tensor.t * Tensor.t