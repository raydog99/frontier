open Torch

type t

val create:
  num_heads:int ->
  head_dim:int ->
  qas_space:(module QASSpace.QAS_SPACE) ->
  t

val forward:
  t ->
  Tensor.t ->          (* query *)
  ?mask:Tensor.t ->    (* optional attention mask *)
  Tensor.t             (* output *)