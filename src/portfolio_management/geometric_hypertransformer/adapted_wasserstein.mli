open Torch

type transport_plan = {
  coupling: Tensor.t;
  marginals: Tensor.t * Tensor.t;
  cost: float;
}

val compute_bicausal_transport:
  Tensor.t ->          (* source measure *)
  Tensor.t ->          (* target measure *)
  filtration:filtration ->
  transport_plan

val adapted_wasserstein_distance:
  int ->               (* p-Wasserstein *)
  transport_plan ->
  float