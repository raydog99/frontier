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

module RiemannianMetrics = struct
  let create_geodesic geometry start_point end_point =
    let velocity t =
      let tangent = Tensor.sub end_point start_point in
      Tensor.mul_scalar tangent (1. -. t)
    in
    
    let acceleration t =
      let v = velocity t in
      let gamma = geometry.InformationGeometry.connection.christoffel 
        (Tensor.add start_point (Tensor.mul_scalar (velocity t) t)) in
      
      (* Compute geodesic acceleration *)
      Array.init geometry.dimension (fun i ->
        Array.init geometry.dimension (fun j ->
          Array.init geometry.dimension (fun k ->
            let coeff = gamma.(i).(j).(k) in
            Tensor.mul coeff (Tensor.mul v v)
          ) |> Array.fold_left Tensor.add (Tensor.zeros [geometry.dimension])
        ) |> Array.fold_left Tensor.add (Tensor.zeros [geometry.dimension])
      ) |> Array.fold_left Tensor.add (Tensor.zeros [geometry.dimension])
    in

    let curve t =
      Tensor.add start_point (Tensor.mul_scalar (velocity t) t)
    in

    let length =
      let metric = geometry.metric.metric_tensor start_point in
      let v = velocity 0. in
      sqrt (Tensor.dot v (Tensor.mv metric v))
    in

    {curve; velocity; acceleration; length}

  let sectional_curvature geometry point u v =
    let riemann = ref 0. in
    let n = geometry.dimension in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        for k = 0 to n - 1 do
          for l = 0 to n - 1 do
            let r_ijkl = ref 0. in
            let gamma = geometry.connection.christoffel point in
            
            for m = 0 to n - 1 do
              r_ijkl := !r_ijkl +.
                Tensor.float_value (Tensor.get gamma.(i).(j).(m)) *.
                Tensor.float_value (Tensor.get gamma.(m).(k).(l))
            done;
            
            riemann := !riemann +. !r_ijkl *.
              Tensor.float_value (Tensor.get u [|i|]) *.
              Tensor.float_value (Tensor.get u [|k|]) *.
              Tensor.float_value (Tensor.get v [|j|]) *.
              Tensor.float_value (Tensor.get v [|l|])
          done
        done
      done
    done;
    
    !riemann /. (Tensor.dot u u *. Tensor.dot v v -. 
                 (Tensor.dot u v) ** 2.)
end

module AdaptedMetrics = struct
  type filtration_metric = {
    base_distance: Tensor.t -> Tensor.t -> float;
    temporal_weight: float -> float -> float;
    causality_weight: int -> float;
  }

  let create_adapted_metric ~base_distance ~decay_rate =
    let temporal_weight s t =
      exp (-. decay_rate *. abs_float (t -. s))
    in
    
    let causality_weight n =
      exp (-. decay_rate *. float_of_int (abs n))
    in
    
    {base_distance; temporal_weight; causality_weight}

  let compute_adapted_distance metric p1 p2 time =
    let d = metric.base_distance p1 p2 in
    let w = metric.temporal_weight time (time +. 1.) in
    d *. w
end