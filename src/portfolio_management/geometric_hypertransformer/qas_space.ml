open Torch

type point = {
  value: Tensor.t;
  metric: Tensor.t -> Tensor.t -> float;
  geometry: InformationGeometry.geometric_structure;
}

type t = point

type mixing_function = {
  mix: float array -> point array -> point;
  constant: float;
  power: int;
}

let create_mixing constant power = {
  mix = (fun weights points ->
    let n = Array.length points in
    
    let rec iterate current_point iter =
      if iter >= 100 then current_point
      else
        let gradients = Array.mapi (fun i p ->
          let v = InformationGeometry.logarithmic_map 
            current_point.geometry current_point.value p.value in
          Tensor.mul_scalar v weights.(i)
        ) points in
        
        let update = Array.fold_left Tensor.add 
          (Tensor.zeros_like current_point.value) gradients in
        
        let next_point = 
          InformationGeometry.exponential_map
            current_point.geometry current_point.value update in
        
        let error = Tensor.norm 
          (Tensor.sub next_point current_point.value) in
        
        if Tensor.float_value error < 1e-6 then
          {current_point with value = next_point}
        else
          iterate {current_point with value = next_point} (iter + 1)
    in
    
    iterate (Array.get points 0) 0
  );
  constant;
  power;
}

let distance p1 p2 = p1.metric p1.value p2.value

let metric_capacity epsilon =
  let log2 x = log x /. log 2. in
  int_of_float (ceil (log2 (1. /. epsilon)))

let quantization_modulus epsilon =
  let log2 x = log x /. log 2. in
  int_of_float (ceil (log2 (1. /. epsilon)))

let mix = fun m -> m.mix

let quantize q tensor = 
  match space_type with
  | Wasserstein -> 
      (* Quantization for Wasserstein space using k-means *)
      let kmeans = KMeans.fit ~n_clusters:q ~points:tensor in
      let centroids = kmeans.centroids in
      let weights = 
        Tensor.ones [q] |> fun x -> 
        Tensor.div_scalar x (float_of_int q) in
      {
        support = centroids;
        weights;
        dimension = Tensor.size tensor (-1)
      }
  | RKHS ->
      (* Quantization for RKHS using feature map *)
      let feature_map x =
        match kernel.feature_map x with
        | Some fm -> fm
        | None -> 
            (* Use kernel trick when explicit feature map unavailable *)
            let gram = kernel.k x x in
            Tensor.matmul gram x
      in
      let features = feature_map tensor in
      let quantized = generic_quantize q features in
      {
        value = quantized;
        kernel = gaussian_kernel (1. /. float_of_int q);
        norm = sqrt (
          Tensor.dot quantized 
            (Tensor.mv (kernel.k quantized quantized) quantized))
      }
  | StatisticalManifold ->
      (* Quantization for statistical manifold *)
      let dim = Tensor.size tensor (-1) in
      let geometry = 
        InformationGeometry.create_statistical_manifold dim in
      let natural_params = Tensor.reshape tensor [-1; dim] in
      let quantized_params = generic_quantize q natural_params in
      let exp_params = 
        InformationGeometry.exponential_map
          geometry
          (Tensor.zeros [dim])
          quantized_params in
      {
        natural_params = quantized_params;
        expectation_params = exp_params;
        geometry;
      }
  | Generic -> 
    let batch_size = Tensor.size tensor 0 in
    let dim = Tensor.size tensor (-1) in
    
    (* Compute quantization grid based on epsilon-net *)
    let epsilon = 1. /. float_of_int q in
    let grid_size = int_of_float (ceil (1. /. epsilon)) in
    
    (* Create uniform grid in [0,1]^d *)
    let make_grid () =
      let points = ref [] in
      let rec generate_point current_dim coords =
        if current_dim = dim then
          points := coords :: !points
        else
          for i = 0 to grid_size - 1 do
            let x = float_of_int i *. epsilon in
            generate_point (current_dim + 1) (x :: coords)
          done
      in
      generate_point 0 [];
      !points
    in
    
    let grid_points = make_grid () in
    let grid_tensor = 
      Tensor.of_float2 (Array.of_list grid_points) in
    
    (* Find nearest grid points for each input point *)
    let quantize_point point =
      let distances = Tensor.cdist 
        (Tensor.reshape point [1; -1]) 
        grid_tensor in
      let _, indices = Tensor.min distances ~dim:1 in
      Tensor.select grid_tensor ~dim:0 
        ~index:(Tensor.int_value indices)
    in
    
    (* Process batch *)
    let quantized = ref [] in
    for i = 0 to batch_size - 1 do
      let point = Tensor.select tensor ~dim:0 ~index:i in
      let quant_point = quantize_point point in
      quantized := quant_point :: !quantized
    done;
    
    (* Stack quantized points *)
    Tensor.stack (List.rev !quantized) ~dim:0

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
  let geometry = point.geometry in
  let projected = 
    InformationGeometry.project_to_model geometry point.value in
  {point with value = projected}

let decompress point = point