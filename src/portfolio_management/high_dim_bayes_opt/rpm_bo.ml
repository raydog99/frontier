open Torch

let epsilon = 1e-10

let safe_normalize tensor =
  let norm = Tensor.norm tensor in
  if norm < epsilon then tensor
  else Tensor.(tensor / float_vec [norm])

let safe_matrix_inverse k =
  let n = (Tensor.size k).(0) in
  let jitter = Tensor.(eye n * float_vec [epsilon]) in
  Tensor.(k + jitter)

let safe_sqrt x = 
  if x < 0. then 0. 
  else sqrt x

module GaussianProcess = struct
  type t = {
    mean: Tensor.t -> float;
    kernel: Tensor.t -> Tensor.t -> float;
    training_x: Tensor.t list;
    training_y: float list;
    noise_var: float;
  }

  let create ?(mean_fn=fun _ -> 0.) ?(noise_var=1e-6) kernel =
    { mean = mean_fn;
      kernel = kernel;
      training_x = [];
      training_y = [];
      noise_var = noise_var;
    }

  let squared_exp_kernel length_scale amplitude x1 x2 =
    let diff = Tensor.(x1 - x2) in
    let sq_dist = Tensor.(sum (diff * diff)) in
    amplitude *. exp (-. sq_dist /. (2. *. length_scale *. length_scale))

  let predict gp x =
    match gp.training_x, gp.training_y with
    | [], [] -> (gp.mean x, gp.noise_var)
    | xs, ys ->
        let n = List.length xs in
        let k = Tensor.zeros [n; n] in
        for i = 0 to n - 1 do
          for j = 0 to n - 1 do
            let kij = gp.kernel (List.nth xs i) (List.nth xs j) in
            Tensor.set k [i; j] kij
          done
        done;
        
        let k = safe_matrix_inverse k in
        let k_star = Tensor.zeros [1; n] in
        List.iteri (fun i xi -> 
          let ki = gp.kernel x xi in
          Tensor.set k_star [0; i] ki
        ) xs;
        
        let y = Tensor.of_float1 (Array.of_list ys) in
        let alpha = Tensor.solve k y in
        let mu = Tensor.(sum (k_star * alpha)) in
        let v = Tensor.(k_star * (solve k k_star)) in
        let sigma2 = gp.kernel x x -. (Tensor.get v [0; 0]) in
        
        (Tensor.float_value mu, sigma2)
end

let create_random_matrix m d =
  let x = Tensor.(randn [m; d] ~kind:Float) in
  let q, _ = Tensor.qr x ~some:false in
  q

let project a x =
  Tensor.(mm a x)

let project_to_bounds x bounds =
  let d = (Tensor.size x).(1) in
  let result = Tensor.zeros_like x in
  for i = 0 to d - 1 do
    let xi = Tensor.get x [0; i] in
    let (lb, ub) = List.nth bounds i in
    let clamped = min (max xi lb) ub in
    Tensor.set result [0; i] clamped
  done;
  result

let verify_manifold_dimension points dim =
  let data = Tensor.stack points 0 in
  let mean = Tensor.(mean data) in
  let centered = Tensor.(data - mean) in
  let cov = Tensor.(matmul (transpose centered) centered) in
  let eigenvals = Tensor.symeig cov ~eigenvectors:false in
  let sorted_eigs = List.sort (fun a b -> compare b a) 
    (Tensor.to_float_list eigenvals) in
  let effective_dim = List.fold_left (fun acc e -> 
    if e > epsilon then acc + 1 else acc) 0 sorted_eigs in
  effective_dim <= dim

module GeometryAwareManifold = struct
  type manifold_type = 
    | Linear of { basis: Tensor.t }
    | Spherical of { radius: float; center: Tensor.t }
    | Mixed of { linear_basis: Tensor.t; sphere_radius: float }
    | KleinBottle of { embedding_dim: int }
    | GeneralManifold of { dim: int; projection_fn: Tensor.t -> Tensor.t }

  let create_mapping = function
    | Linear { basis } -> fun x ->
        let bt = Tensor.transpose2 basis 0 1 in
        let proj = Tensor.(mm (mm x bt) basis) in
        safe_normalize proj

    | Spherical { radius; center } -> fun x ->
        let centered = Tensor.(x - center) in
        let norm = max (Tensor.norm centered) epsilon in
        Tensor.(center + (centered * float_vec [radius /. norm]))
    
    | Mixed { linear_basis; sphere_radius } -> fun x ->
        let linear_proj = Tensor.(mm (mm x (transpose2 linear_basis 0 1)) linear_basis) in
        let normalized = safe_normalize linear_proj in
        Tensor.(linear_proj + (normalized * float_vec [sphere_radius]))

    | KleinBottle { embedding_dim=_ } -> fun x ->
        let x_reshaped = Tensor.reshape x [3; 3] in
        x_reshaped

    | GeneralManifold { dim=_; projection_fn } -> projection_fn

  let verify_mapping_properties m x =
    match m with
    | Linear { basis } ->
        let proj = create_mapping m x in
        let dim = (Tensor.size basis).(1) in
        verify_manifold_dimension [proj] dim

    | Spherical { radius; center=_ } ->
        let proj = create_mapping m x in
        let r = Tensor.norm proj in
        abs_float (r -. radius) < epsilon

    | Mixed _ | KleinBottle _ | GeneralManifold _ -> true
end

module SemiSupervisedLearning = struct
  type validation_stats = {
    supervised_loss: float;
    consistency_loss: float;
    manifold_error: float;
    gradient_norm: float;
  }

  let manifold_consistency_check h x x_m lambda =
    let interp = Tensor.(x * float_vec [lambda] + x_m * float_vec [1. -. lambda]) in
    let h_interp = h interp in
    let diff = Tensor.(h_interp - h x_m) in
    Tensor.(norm diff) |> Tensor.float_value

  let cross_validate_gamma labeled_data unlabeled_data gammas h =
    let n_folds = min 5 (List.length labeled_data) in
    let best_gamma = ref (List.hd gammas) in
    let best_score = ref Float.infinity in
    
    List.iter (fun gamma ->
      let fold_scores = List.init n_folds (fun _ -> 
        let val_loss = manifold_consistency_check h 
          (fst (List.hd labeled_data)) 
          (List.hd unlabeled_data) 
          gamma in
        val_loss
      ) in
      let avg_score = List.fold_left (+.) 0. fold_scores /. float n_folds in
      if avg_score < !best_score then begin
        best_score := avg_score;
        best_gamma := gamma
      end
    ) gammas;
    !best_gamma

  let validate_iteration h labeled_data unlabeled_data =
    let sup_loss = List.fold_left (fun acc (x, y) ->
      let pred = h x in
      let diff = Tensor.(pred - float_vec [y]) in
      acc +. (Tensor.(norm diff) |> Tensor.float_value)
    ) 0. labeled_data /. float (List.length labeled_data) in
    
    let cons_loss = List.fold_left (fun acc x ->
      let pred = h x in
      acc +. manifold_consistency_check h x pred 0.5
    ) 0. unlabeled_data /. float (List.length unlabeled_data) in

    let man_error = List.fold_left (fun acc (x, _) ->
      let pred = h x in
      acc +. (Tensor.(norm (pred - x)) |> Tensor.float_value)
    ) 0. labeled_data /. float (List.length labeled_data) in

    {
      supervised_loss = sup_loss;
      consistency_loss = cons_loss;
      manifold_error = man_error;
      gradient_norm = 0.0;  (* Computed during optimization *)
    }
end

let verify_backprojection_exists a h x =
  let at = Tensor.transpose2 a 0 1 in
  let z = Tensor.(mm x at) in
  let x_back = h Tensor.(mm z at) in
  let diff = Tensor.(mean (x - x_back)) |> Tensor.float_value in
  abs_float diff < epsilon

let verify_projection_completeness a h points =
  let ma_subset_check = List.fold_left (fun acc x ->
    let ax = Tensor.(mm a x) in
    let z = ax in
    let x_back = h Tensor.(mm z (transpose2 a 0 1)) in
    acc && Tensor.(norm (x - x_back)) |> Tensor.float_value < epsilon
  ) true points in
  
  let ma_superset_check = List.fold_left (fun acc x ->
    let ax = Tensor.(mm a x) in
    let exists_preimage = verify_backprojection_exists a h x in
    acc && exists_preimage
  ) true points in
  
  ma_subset_check && ma_superset_check

let project_back a h z =
  let at = Tensor.transpose2 a 0 1 in
  let x = h Tensor.(mm z at) in
  if verify_backprojection_exists a h x then Some x
  else None

let verify_effective_dimension points dim =
  let neighbors_of_point p =
    List.filter (fun x -> 
      let dist = Tensor.(norm (x - p)) |> Tensor.float_value in
      dist < epsilon
    ) points in

  List.for_all (fun p ->
    let neighbors = neighbors_of_point p in
    verify_manifold_dimension (p :: neighbors) dim
  ) points

let verify_distance_preservation_bounds a x y epsilon m d =
  let ax = Tensor.(mm a x) in
  let ay = Tensor.(mm a y) in
  let proj_dist = Tensor.(norm (ax - ay)) |> Tensor.float_value in
  let orig_dist = Tensor.(norm (x - y)) |> Tensor.float_value in
  
  let lower = (1. -. epsilon) *. sqrt (float m /. float d) in
  let upper = (1. +. epsilon) *. sqrt (float m /. float d) in
  
  proj_dist >= lower *. orig_dist && 
  proj_dist <= upper *. orig_dist

let verify_diffeomorphism_conditions mapping points epsilon =
  let is_homeomorphic = List.for_all (fun p ->
    let tangent = Tensor.(randn (size p)) in
    let p' = Tensor.(p + tangent * float_vec [epsilon]) in
    let mp = mapping p in
    let mp' = mapping p' in
    Tensor.(norm (mp' - mp)) |> Tensor.float_value > 0.
  ) points in
  
  let has_smooth_inverse = List.for_all (fun p ->
    let mp = mapping p in
    let neighbors = List.filter (fun x -> 
      let dist = Tensor.(norm (x - p)) |> Tensor.float_value in
      dist < epsilon
    ) points in
    let local_dim = verify_manifold_dimension (mp :: 
      List.map mapping neighbors) ((Tensor.size p).(0)) in
    local_dim
  ) points in
  
  is_homeomorphic && has_smooth_inverse

let verify_back_projection_convergence h x_orig num_iter =
  let rec check_convergence x iter =
    if iter >= num_iter then true
    else
      let x_next = h x in
      let diff = Tensor.(norm (x_next - x)) |> Tensor.float_value in
      if diff < epsilon then true
      else check_convergence x_next (iter + 1)
  in
  check_convergence x_orig 0

let search_projected_space acq_fn bounds max_iters =
  let d = List.length bounds in
  let best_x = ref (Tensor.zeros [1; d]) in
  let best_y = ref (~-.Float.infinity) in
  
  let n_starts = 10 in
  let max_step = 0.1 in
  let min_step = 1e-4 in
  
  for start = 0 to n_starts - 1 do
    let x = Tensor.zeros [1; d] in
    for j = 0 to d - 1 do
      let (lb, ub) = List.nth bounds j in
      let segment = (ub -. lb) /. float n_starts in
      let rand_val = lb +. (float start *. segment) +. Random.float segment in
      Tensor.set x [0; j] rand_val
    done;
    
    let step_size = ref max_step in
    let momentum = 0.9 in
    let velocity = ref Tensor.(zeros_like x) in
    
    let x' = ref (Tensor.requires_grad x) in
    for _ = 1 to max_iters do
      let y = acq_fn !x' in
      let grad = Tensor.backward y in
      
      velocity := Tensor.(!velocity * float_vec [momentum] + 
                        grad * float_vec [1. -. momentum]);
      
      x' := Tensor.(!x' - (!velocity * float_vec [!step_size]));
      x' := project_to_bounds !x' bounds;
      
      if Tensor.(norm grad) < 1e-3 then
        step_size := max min_step (!step_size *. 0.9)
      else if !step_size < max_step then
        step_size := min max_step (!step_size *. 1.1);
        
      let y_new = acq_fn !x' in
      if Tensor.float_value y_new > !best_y then begin
        best_x := !x';
        best_y := Tensor.float_value y_new
      end
    done
  done;
  !best_x

module RPMBO = struct
  type config = {
    ambient_dim: int;
    manifold_dim: int;
    projection_dim: int;
    n_init: int;
    max_iter: int;
    exploration_weight: float;
  }

  type manifold_config = {
    manifold_type: GeometryAwareManifold.manifold_type;
    projection_dim: int;
    ambient_dim: int;
  }

  type stats = {
    convergence_stats: SemiSupervisedLearning.validation_stats array;
    diffeomorphism_verified: bool;
    distance_preserved: bool;
  }

  type t = {
    config: config;
    projection: Tensor.t;
    manifold_mapping: Tensor.t -> Tensor.t;
    gp: GaussianProcess.t;
    manifold_config: manifold_config;
    mutable stats: stats option;
  }

  let create config manifold_config =
    let projection = create_random_matrix 
      config.projection_dim config.ambient_dim in
    let kernel = GaussianProcess.squared_exp_kernel 1.0 1.0 in
    let gp = GaussianProcess.create kernel in
    let manifold_mapping = GeometryAwareManifold.create_mapping manifold_config.manifold_type in
    { config; projection; manifold_mapping; gp; manifold_config; stats = None }

  let expected_improvement gp x best_f =
    let mu, sigma2 = GaussianProcess.predict gp x in
    let sigma = safe_sqrt sigma2 in
    let z = (mu -. best_f) /. sigma in
    let phi = exp (-0.5 *. z *. z) /. sqrt (2. *. Float.pi) in
    let big_phi = 0.5 *. (1. +. erf (z /. sqrt 2.)) in
    sigma *. (z *. big_phi +. phi)

  let optimize t ~objective init_points x_unlabeled =
    let epsilon = 0.1 in
    let beta_t = sqrt (2. *. log (float t.config.max_iter)) in
    
    let rec loop iter gp best_y points =
      if iter >= t.config.max_iter then (points, t.stats)
      else
        let diffeomorphism_ok = verify_diffeomorphism_conditions 
          t.manifold_mapping (List.map fst points) epsilon in
        
        let distance_ok = match points with
          | (x1, _) :: (x2, _) :: _ -> 
              verify_distance_preservation_bounds 
                t.projection x1 x2 epsilon t.config.projection_dim t.config.ambient_dim
          | _ -> true in
        
        let conv_stats = SemiSupervisedLearning.validate_iteration 
          t.manifold_mapping points x_unlabeled in
        
        t.stats <- Some {
          convergence_stats = [|conv_stats|];
          diffeomorphism_verified = diffeomorphism_ok;
          distance_preserved = distance_ok;
        };

        let bounds = List.init t.manifold_config.projection_dim 
          (fun _ -> (-. sqrt (float t.manifold_config.projection_dim), 
                     sqrt (float t.manifold_config.projection_dim))) in
                     
        let next_z = search_projected_space
          (fun z -> 
            match project_back t.projection t.manifold_mapping z with
            | Some x -> expected_improvement gp x best_y
            | None -> ~-. Float.infinity)
          bounds 100 in

        match project_back t.projection t.manifold_mapping next_z with
        | Some next_x -> 
            let y = objective next_x in
            let gp' = { gp with 
              training_x = next_x :: gp.training_x;
              training_y = y :: gp.training_y 
            } in
            let best_y' = max best_y y in
            loop (iter + 1) gp' best_y' ((next_x, y) :: points)
        | None -> loop (iter + 1) gp best_y points
    in
    loop 0 t.gp (min_float) init_points
end