val positive_definite_projection: Tensor.t -> Tensor.t

val parallel_transport_tensor: 
  Tensor.t ->        (* start_point *)
  Tensor.t ->        (* end_point *)
  Tensor.t ->        (* vector *)
  (Tensor.t -> Tensor.t array array array) ->  (* christoffel *)
  Tensor.t

val geodesic_distance: 
  (Tensor.t -> Tensor.t) ->  (* metric tensor *)
  Tensor.t ->                (* point1 *)
  Tensor.t ->                (* point2 *)
  float