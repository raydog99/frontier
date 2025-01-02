open Torch

type point = {
  natural_params: Tensor.t;    (* θ coordinates *)
  expectation_params: Tensor.t; (* η coordinates *)
  geometry: InformationGeometry.statistical_manifold;
}

type t = point

type mixing_function = {
  mix: float array -> point array -> point;
  constant: float;
  power: int;
}

let create_mixing constant power =
  let mix weights points =
    (* Compute mixture in exponential family *)
    let compute_mixture points weights =
      let dim = (Array.get points 0).geometry.dimension in
      let total_params = Tensor.zeros [dim] in
      
      Array.iteri (fun i p ->
        let weighted_params = 
          Tensor.mul_scalar p.natural_params weights.(i) in
        Tensor.add_ total_params weighted_params
      ) points;

      (* Project back to manifold *)
      let geometry = (Array.get points 0).geometry in
      let exp_params = 
        InformationGeometry.exponential_map
          geometry
          (Array.get points 0).natural_params
          total_params in

      {
        natural_params = total_params;
        expectation_params = exp_params;
        geometry = (Array.get points 0).geometry;
      }
    in
    compute_mixture points weights
  in
  {mix; constant; power}

let distance p1 p2 =
  let metric = p1.geometry.metric.metric_tensor p1.natural_params in
  let diff = Tensor.sub p2.natural_params p1.natural_params in
  sqrt (Tensor.dot diff (Tensor.mv metric diff))

let metric_capacity epsilon =
  let log2 x = log x /. log 2. in
  int_of_float (ceil (log2 (1. /. epsilon)))

let quantization_modulus epsilon =
  let log2 x = log x /. log 2. in
  int_of_float (ceil (log2 (1. /. epsilon)))

let mix = fun m -> m.mix

let quantize q tensor =
  let dim = Tensor.size tensor (-1) in
  let geometry = 
    InformationGeometry.create_statistical_manifold dim in
  
  (* Project to statistical manifold *)
  let natural_params = 
    Tensor.reshape tensor [-1; dim] in
  let exp_params = 
    InformationGeometry.exponential_map
      geometry
      (Tensor.zeros [dim])
      natural_params in
  
  {
    natural_params;
    expectation_params = exp_params;
    geometry;
  }

let verify_simplicial mixing points =
  Array.for_all (fun point ->
    let mixed = mixing.mix 
      (Array.make (Array.length points) 
         (1. /. float_of_int (Array.length points))) 
      points in
    distance mixed point <= 
      mixing.constant *. 
      (Array.fold_left (fun acc p ->
         max acc (distance point p)
       ) 0. points) ** float_of_int mixing.power
  ) points

let compress point epsilon =
  let dim = point.geometry.dimension in
  let reduced_dim = 
    max 1 (int_of_float (ceil (1. /. epsilon))) in
  
  (* Project to lower dimensional manifold *)
  let projection = 
    Tensor.svd point.natural_params 
    |> fun (u, s, _) ->
       Tensor.narrow u ~dim:1 ~start:0 ~length:reduced_dim in
  
  let reduced_params = 
    Tensor.matmul projection point.natural_params in
  
  let geometry = 
    InformationGeometry.create_statistical_manifold reduced_dim in
  
  {
    natural_params = reduced_params;
    expectation_params = 
      InformationGeometry.exponential_map
        geometry
        (Tensor.zeros [reduced_dim])
        reduced_params;
    geometry;
  }

let decompress point = point