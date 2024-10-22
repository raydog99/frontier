open Torch

module AdaptedWassersteinMetric : sig
  val create_metric: float -> MetricStructures.AdaptedMetrics.filtration_metric
  val distance: MetricStructures.AdaptedMetrics.filtration_metric -> 
               Tensor.t -> Tensor.t -> float -> float
end

module RKHSMetric : sig
  val gaussian_kernel: float -> Tensor.t -> Tensor.t -> float
  val create_metric: float -> MetricStructures.AdaptedMetrics.filtration_metric
end

module StatisticalManifoldMetric : sig
  val create_metric: int -> MetricStructures.AdaptedMetrics.filtration_metric
end