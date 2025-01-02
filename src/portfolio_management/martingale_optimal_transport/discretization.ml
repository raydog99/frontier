open Torch

type discretization_params = {
  theta: float;
  m_theta: float;
  dimension: int;
  truncation_radius: float;
}

type quantization_method = 
  | Uniform
  | Lloyd
  | PrincipalCurve
  | OptimalVoronoi

let create_params ~theta ~m_theta ~dimension ~truncation_radius =
  if theta <= 1.0 then failwith "Theta must be greater than 1";
  if dimension < 1 then failwith "Dimension must be positive";
  {theta; m_theta; dimension; truncation_radius}

let lloyd_quantization measure n params =
  let support = DiscreteMeasure.support measure in
  let density = DiscreteMeasure.density measure |> Option.get in
  
  (* Initialize *)
  let min_val = Tensor.min support |> Tensor.float_value in
  let max_val = Tensor.max support |> Tensor.float_value in
  let centroids = Tensor.rand [|n; params.dimension|] |>
    Tensor.mul_scalar (max_val -. min_val) |>
    Tensor.add_scalar min_val in
  
  let rec iterate centroids iter =
    if iter >= 100 then centroids
    else
      (* Assign points *)
      let distances = Tensor.cdist support centroids ~p:2.0 in
      let assignments = Tensor.argmin distances ~dim:1 ~keepdim:false in
      
      (* Update *)
      let new_centroids = Tensor.zeros_like centroids in
      let counts = Tensor.zeros [|n|] in
      
      Tensor.iteri (fun idx _ ->
        let centroid_idx = Tensor.item assignments idx.(0) in
        let point = Tensor.narrow support ~dim:0 ~start:idx.(0) ~length:1 in
        let weight = Tensor.item density idx.(0) in
        
        Tensor.add_ (Tensor.narrow new_centroids ~dim:0 ~start:centroid_idx ~length:1)
          (Tensor.mul_scalar point weight);
        Tensor.add_ (Tensor.narrow counts ~dim:0 ~start:centroid_idx ~length:1)
          (Tensor.scalar_tensor weight)
      ) assignments;
      
      let normalized = Tensor.div new_centroids 
        (Tensor.reshape counts [|n; 1|]) in
      
      let diff = Tensor.norm 
        (Tensor.sub normalized centroids) 
        ~p:2 ~dim:[|0; 1|] |>
        Tensor.float_value in
      
      if diff < 1e-6 then normalized
      else iterate normalized (iter + 1)
  in
  
  let final_centroids = iterate centroids 0 in
  let final_weights = Tensor.ones [|n|] |>
    Tensor.div_scalar (float_of_int n) in
  
  DiscreteMeasure.create final_weights ~support:final_centroids

let principal_curve_quantization measure n params =
  let support = DiscreteMeasure.support measure in
  let density = DiscreteMeasure.density measure |> Option.get in
  
  (* Initial PCA direction *)
  let centered = Tensor.sub support 
    (Tensor.mean support ~dim:[|0|]) in
  let cov = Tensor.matmul 
    (Tensor.transpose centered ~dim0:0 ~dim1:1) 
    centered in
  let eigvals, eigvecs = Tensor.symeig cov ~eigenvectors:true in
  
  (* Initialize curve *)
  let init_direction = Tensor.select eigvecs ~dim:1 ~index:0 in
  let curve_points = Tensor.linspace (-1.0) 1.0 n |>
    Tensor.reshape [|n; 1|] |>
    Tensor.matmul init_direction in
  
  let rec iterate curve iter =
    if iter >= 100 then curve
    else
      (* Project points onto curve *)
      let distances = Tensor.cdist support curve ~p:2.0 in
      let projections = Tensor.argmin distances ~dim:1 ~keepdim:false in
      
      (* Update curve points *)
      let new_curve = Tensor.zeros_like curve in
      let counts = Tensor.zeros [|n|] in
      
      Tensor.iteri (fun idx _ ->
        let proj_idx = Tensor.item projections idx.(0) in
        let point = Tensor.narrow support ~dim:0 ~start:idx.(0) ~length:1 in
        let weight = Tensor.item density idx.(0) in
        
        Tensor.add_ (Tensor.narrow new_curve ~dim:0 ~start:proj_idx ~length:1)
          (Tensor.mul_scalar point weight);
        Tensor.add_ (Tensor.narrow counts ~dim:0 ~start:proj_idx ~length:1)
          (Tensor.scalar_tensor weight)
      ) projections;
      
      let normalized = Tensor.div new_curve 
        (Tensor.reshape counts [|n; 1|]) in
      
      let diff = Tensor.norm 
        (Tensor.sub normalized curve) 
        ~p:2 ~dim:[|0; 1|] |>
        Tensor.float_value in
      
      if diff < 1e-6 then normalized
      else iterate normalized (iter + 1)
  in
  
  let final_curve = iterate curve_points 0 in
  let final_weights = Tensor.ones [|n|] |>
    Tensor.div_scalar (float_of_int n) in
  
  DiscreteMeasure.create final_weights ~support:final_curve

let discretize measure n params method_type =
  match method_type with
  | Uniform ->
      let support = DiscreteMeasure.support measure in
      let min_val = Tensor.min support |> Tensor.float_value in
      let max_val = Tensor.max support |> Tensor.float_value in
      let points = Tensor.linspace min_val max_val n in
      let weights = Tensor.ones [|n|] |>
        Tensor.div_scalar (float_of_int n) in
      DiscreteMeasure.create weights ~support:points
  | Lloyd -> lloyd_quantization measure n params
  | PrincipalCurve -> principal_curve_quantization measure n params
  | OptimalVoronoi -> 
      lloyd_quantization measure n params

let compute_error original discretized =
  DiscreteMeasure.wasserstein_distance original discretized

module RandomDiscretization = struct
  type sampling_method =
    | Pure
    | Stratified
    | ImportanceBased
    | QuasiMonteCarlo

  type sampling_params = {
    method_type: sampling_method;
    dimension: int;
    confidence_level: float;
    seed: int option;
  }

  (* Sobol sequence generator *)
  module Sobol = struct
    type direction_numbers = {
      dimension: int;
      primitive_polynomials: int array;
      direction_numbers: int array array;
    }

    let init_direction_numbers dim =
      let primitive_polynomials = [|
        1; 3; 7; 11; 13; 19; 25; 37; 59; 47;
        61; 55; 41; 67; 97; 91; 109; 103; 115; 131
      |] in
      
      let direction_numbers = Array.make_matrix dim 32 0 in
      for d = 0 to min (dim-1) 19 do
        let rec init_dim m v =
          if m >= 32 then ()
          else begin
            let v_next = if m = 0 then
              1 lsl (31 - m)
            else
              let p = primitive_polynomials.(d) in
              let term = ref 0 in
              for i = 1 to Int.floor_log2 p do
                if p land (1 lsl i) <> 0 then
                  term := !term lxor (v lsl i)
              done;
              v lxor (!term lsr 1)
            in
            direction_numbers.(d).(m) <- v_next;
            init_dim (m+1) v_next
          end
        in
        init_dim 0 1
      done;
      {dimension = dim; 
       primitive_polynomials = Array.sub primitive_polynomials 0 dim;
       direction_numbers}

    let next_point dirs index =
      let point = Array.make dirs.dimension 0.0 in
      for d = 0 to dirs.dimension - 1 do
        let gray = index lxor (index lsr 1) in
        let value = ref 0 in
        for b = 0 to 31 do
          if gray land (1 lsl b) <> 0 then
            value := !value lxor dirs.direction_numbers.(d).(b)
        done;
        point.(d) <- float_of_int !value /. 2.0 ** 32.0
      done;
      Tensor.of_float1 point
  end

  let stratified_sample measure n params =
    let support = DiscreteMeasure.support measure in
    let density = DiscreteMeasure.density measure |> Option.get in
    
    (* Divide support into strata *)
    let sorted_support, indices = Tensor.sort support ~dim:0 in
    let stride = max 1 ((Tensor.shape support).(0) / n) in
    
    let strata_bounds = Array.init (n+1) (fun i ->
      if i = 0 then Tensor.min support |> Tensor.float_value
      else if i = n then Tensor.max support |> Tensor.float_value
      else Tensor.item sorted_support (min ((Tensor.shape support).(0)-1) (i * stride))) in
    
    (* Sample from each stratum *)
    let samples = Array.init n (fun i ->
      let lower = strata_bounds.(i) in
      let upper = strata_bounds.(i+1) in
      let stratum_mask = Tensor.logical_and
        (Tensor.ge support (Tensor.scalar_tensor lower))
        (Tensor.lt support (Tensor.scalar_tensor upper)) in
      
      let stratum_support = Tensor.masked_select support stratum_mask in
      let stratum_density = Tensor.masked_select density stratum_mask in
      
      if Tensor.numel stratum_support = 0 then
        Tensor.scalar_tensor ((lower +. upper) /. 2.0)
      else
        let idx = Tensor.multinomial stratum_density ~num_samples:1 ~replacement:true in
        Tensor.index_select stratum_support ~dim:0 ~index:idx) in
    
    let sampled_points = Tensor.stack ~dim:0 (Array.to_list samples) in
    let weights = Tensor.ones [|n|] |> Tensor.div_scalar (float_of_int n) in
    
    (sampled_points, weights)

  let importance_sample measure n params =
    let support = DiscreteMeasure.support measure in
    let density = DiscreteMeasure.density measure |> Option.get in
    
    let importance_weights = 
      let grad = Tensor.sub (Tensor.roll density 1) density in
      let abs_grad = Tensor.abs grad in
      Tensor.add abs_grad (Tensor.mean abs_grad) in
    
    let normalized_weights = Tensor.div importance_weights 
      (Tensor.sum importance_weights) in
    
    let indices = Tensor.multinomial normalized_weights 
      ~num_samples:n ~replacement:false in
    let samples = Tensor.index_select support ~dim:0 ~index:indices in
    
    let sample_weights = Tensor.index_select density ~dim:0 ~index:indices in
    let final_weights = Tensor.div sample_weights 
      (Tensor.sum sample_weights) in
    
    (samples, final_weights)

  let sample measure n params =
    match params.method_type with
    | Pure -> 
        let samples = DiscreteMeasure.sample measure n in
        let weights = Tensor.ones [|n|] |> 
          Tensor.div_scalar (float_of_int n) in
        (samples, weights)
    | Stratified -> stratified_sample measure n params
    | ImportanceBased -> importance_sample measure n params
    | QuasiMonteCarlo ->
        let dirs = Sobol.init_direction_numbers params.dimension in
        let samples = Array.init n (fun i -> 
          Sobol.next_point dirs (i + 1)) in
        let points = Tensor.stack ~dim:0 (Array.to_list samples) in
        let weights = Tensor.ones [|n|] |> 
          Tensor.div_scalar (float_of_int n) in
        (points, weights)

  let estimate_error original sampled params =
    let wasserstein = DiscreteMeasure.wasserstein_distance 
      original sampled in
    
    (* Confidence interval *)
    let z_score = match params.confidence_level with
      | cl when cl >= 0.99 -> 2.576
      | cl when cl >= 0.95 -> 1.96
      | cl when cl >= 0.90 -> 1.645
      | _ -> 1.28 in
    
    let std = match params.method_type with
      | Pure -> wasserstein /. sqrt (float_of_int 
          (Tensor.shape (DiscreteMeasure.support sampled)).(0))
      | Stratified -> wasserstein /. float_of_int 
          (Tensor.shape (DiscreteMeasure.support sampled)).(0)
      | ImportanceBased -> 
          let (_, weights) = importance_sample original 
            (Tensor.shape (DiscreteMeasure.support sampled)).(0) 
            params in
          let var = Tensor.var weights |> Tensor.float_value in
          sqrt var /. sqrt (float_of_int 
            (Tensor.shape (DiscreteMeasure.support sampled)).(0))
      | QuasiMonteCarlo -> wasserstein /. 
          (float_of_int (Tensor.shape 
            (DiscreteMeasure.support sampled)).(0) ** 0.5) in
    
    let margin = z_score *. std in
    let ci_lower = max 0.0 (wasserstein -. margin) in
    let ci_upper = wasserstein +. margin in
    
    (wasserstein, (ci_lower, ci_upper))
end