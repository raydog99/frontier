open Torch

val retract :
  point:Tensor.t ->
  direction:Tensor.t ->
  Tensor.t
(** Retraction map for positive definite matrices *)

val vector_transport :
  point:Tensor.t ->
  direction:Tensor.t ->
  Tensor.t
(** Vector transport on manifold *)

val optimize :
  objective:(Tensor.t -> float) ->
  gradient:(Tensor.t -> Tensor.t) ->
  init:Tensor.t ->
  params:Types.riemannian_params ->
  Tensor.t
(** Riemannian optimization *)