open Torch

type point = {
  natural_params: Tensor.t;
  expectation_params: Tensor.t;
  geometry: InformationGeometry.statistical_manifold;
}

include QAS_SPACE with type t = point

val create_statistical_manifold: int -> InformationGeometry.statistical_manifold
val create_mixing: float -> int -> mixing_function
val exponential_map: point -> point -> float -> point
val parallel_transport: point -> point -> point -> point