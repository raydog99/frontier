open Torch

type ('a, 'b) metric_morphism = {
  forward: 'a -> 'b;
  backward: 'b -> 'a option;
  distortion: float;
}

type 'a geodesic = {
  curve: float -> 'a;
  velocity: float -> Tensor.t;
  acceleration: float -> Tensor.t;
  length: float;
}

module RiemannianMetrics : sig
  val create_geodesic: 
    InformationGeometry.statistical_manifold -> 
    Tensor.t -> 
    Tensor.t -> 
    Tensor.t geodesic

  val sectional_curvature: 
    InformationGeometry.statistical_manifold -> 
    Tensor.t -> 
    Tensor.t -> 
    Tensor.t -> 
    float
end

module AdaptedMetrics : sig
  type filtration_metric = {
    base_distance: Tensor.t -> Tensor.t -> float;
    temporal_weight: float -> float -> float;
    causality_weight: int -> float;
  }

  val create_adapted_metric:
    base_distance:(Tensor.t -> Tensor.t -> float) ->
    decay_rate:float ->
    filtration_metric

  val compute_adapted_distance:
    filtration_metric ->
    Tensor.t ->
    Tensor.t ->
    float ->
    float
end