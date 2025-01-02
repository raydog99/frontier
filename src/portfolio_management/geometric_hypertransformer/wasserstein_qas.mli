open Torch

type point = {
  support: Tensor.t;      (* [batch x points x dim] *)
  weights: Tensor.t;      (* [batch x points] *)
  dimension: int;
}

include QAS_SPACE with type t = point

val create_mixing: float -> int -> mixing_function

val compute_optimal_transport:
  point ->        (* source *)
  point ->        (* target *)
  float ->        (* epsilon *)
  int ->          (* max_iterations *)
  transport_plan  (* result *)

val barycenter:
  weights:float array ->
  points:point array ->
  float ->        (* epsilon *)
  int ->          (* max_iterations *)
  point

and transport_plan = {
  coupling: Tensor.t;
  cost: float;
  source: point;
  target: point;
}