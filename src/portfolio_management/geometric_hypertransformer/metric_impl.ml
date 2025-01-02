open Torch

module AdaptedWassersteinMetric = struct
  let create_metric decay_rate =
    MetricStructures.AdaptedMetrics.create_adapted_metric
      ~base_distance:(fun x y ->
        let cost = Tensor.cdist x y ~p:2. in
        Tensor.mean cost |> Tensor.float_value)
      ~decay_rate

  let distance metric p1 p2 time =
    MetricStructures.AdaptedMetrics.compute_adapted_distance
      metric p1 p2 time
end

module RKHSMetric = struct
  let gaussian_kernel sigma x y =
    let diff = Tensor.sub x y in
    let sq_dist = Tensor.dot diff diff in
    exp (-. Tensor.float_value sq_dist /. 
         (2. *. sigma *. sigma))

  let create_metric sigma =
    let base_distance x y =
      sqrt (
        gaussian_kernel sigma x x +.
        gaussian_kernel sigma y y -.
        2. *. gaussian_kernel sigma x y)
    in
    
    MetricStructures.AdaptedMetrics.create_adapted_metric
      ~base_distance
      ~decay_rate:(1. /. sigma)
end

module StatisticalManifoldMetric = struct
  let create_metric dim =
    let manifold = 
      InformationGeometry.create_statistical_manifold dim in
    
    let base_distance x y =
      TensorUtils.geodesic_distance
        manifold.metric.metric_tensor x y
    in
    
    MetricStructures.AdaptedMetrics.create_adapted_metric
      ~base_distance
      ~decay_rate:1.0
end