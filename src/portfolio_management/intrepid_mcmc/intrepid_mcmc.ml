open Torch

type proposal_config = {
  exploration_ratio : float;
  n_dims : int;
  anchor_point : Tensor.t;
  proposal_scale : float;
  angular_scales : float array;
  radial_gamma : float;
  min_accept_rate : float;
  max_accept_rate : float;
}

type mcmc_state = {
  current_point : Tensor.t;
  current_value : float;
  accepted : bool;
  acceptance_count : int;
  total_count : int;
  parent_density : float;
  transformed_value : float;
}

type distribution_shape =
  | Radially_Symmetric
  | Uniform_Shape of float * float
  | Unimodal_Convex
  | General_Shape

type density_type =
  | Parent
  | Target
  | Transformation

type proposal_type =
  | Local of float
  | Global of float * float array
  | Adaptive of float * float

type sampling_stats = {
  acceptance_rate : float;
  effective_sample_size : float;
  mean : Tensor.t;
  covariance : Tensor.t;
}

type chain_config = {
  n_samples : int;
  n_chains : int;
  n_dims : int;
  burn_in : int;
  thin : int;
  adaptation_window : int;
}

let standard_normal x =
  let norm = Tensor.norm x ~p:2 ~dim:[0] ~keepdim:false in
  let exp_term = Tensor.(-(norm * norm) / (float_value 2.0)) in
  Tensor.(exp exp_term / (float_value (sqrt (2.0 *. Float.pi))))

let multivariate_normal ?(mu=None) ?(sigma=None) x =
  let d = Tensor.size x |> List.hd in
  let mu = match mu with
    | None -> Tensor.zeros [d]
    | Some m -> m
  in
  let sigma = match sigma with
    | None -> Tensor.eye d
    | Some s -> s
  in
  let diff = Tensor.(x - mu) in
  let sigma_inv = Tensor.inverse sigma in
  let quad_form = Tensor.(matmul (view diff ~size:[-1; 1] |> transpose) 
                         (matmul sigma_inv (view diff ~size:[-1; 1])))
                 |> Tensor.squeeze in
  let det = Tensor.det sigma in
  let log_pdf = Tensor.(
    -(float_value (float d *. Float.log (2.0 *. Float.pi)) + log det + quad_form) 
    / (float_value 2.0)
  ) in
  Tensor.exp log_pdf

let uniform ?(lower=None) ?(upper=None) x =
  let d = Tensor.size x |> List.hd in
  let lower = match lower with
    | None -> Tensor.ones [d] |> Tensor.mul_scalar (-1.0)
    | Some l -> l
  in
  let upper = match upper with
    | None -> Tensor.ones [d]
    | Some u -> u
  in
  let within_bounds = Tensor.(
    logical_and (x >= lower) (x <= upper)
    |> all ~dim:[0] ~keepdim:false
  ) in
  let volume = Tensor.(prod (upper - lower)) |> Tensor.float_value in
  if Tensor.to_bool0_int within_bounds then
    1.0 /. volume
  else
    0.0

let mixture weights densities x =
  List.map2 (fun w f -> Tensor.(f x * float_value w)) weights densities
  |> List.fold_left Tensor.(+) (Tensor.zeros [1])
    
let gumbel ?(mu=0.0) ?(beta=1.0) x =
  let z = Tensor.((x - float_value mu) / float_value beta) in
  let exp_term = Tensor.(exp (-(z + exp (neg z)))) in
  Tensor.(exp_term / (float_value beta))

let to_spherical x anchor_point =
  let diff = Tensor.(x - anchor_point) in
  let r = Tensor.norm diff ~p:2 ~dim:[0] ~keepdim:true in
  let normalized = Tensor.(diff / r) in
  (r, normalized)

let from_spherical r theta anchor_point =
  Tensor.(anchor_point + (r * theta))

let log_uniform () =
  Tensor.uniform_float ~from:0. ~to:1. ~size:[1]
  |> Tensor.log

let effective_sample_size chain =
  let n = List.length chain in
  let mean = List.fold_left Tensor.(+) (Tensor.zeros [1]) chain
             |> fun x -> Tensor.(x / float_value (float n)) in
  let auto_corr = Array.make 50 0.0 in
  for lag = 0 to 49 do
    let sum_corr = ref 0.0 in
    for i = 0 to n - lag - 1 do
      let x_i = List.nth chain i in
      let x_lag = List.nth chain (i + lag) in
      sum_corr := !sum_corr +.
        (Tensor.float_value Tensor.((x_i - mean) * (x_lag - mean)));
    done;
    auto_corr.(lag) <- !sum_corr /. (float (n - lag))
  done;
  let tau = 1.0 +. 2.0 *. (Array.fold_left (+.) 0.0 auto_corr) in
  float n /. tau

let compute_stats chain =
  let n = List.length chain in
  let n_float = float n in
  let mean = List.fold_left Tensor.(+) (Tensor.zeros [1]) chain
             |> fun x -> Tensor.(x / float_value n_float) in
  let cov_sum = List.fold_left (fun acc x ->
    let diff = Tensor.(x - mean) in
    let outer = Tensor.(matmul (view diff ~size:[-1; 1]) 
                       (view diff ~size:[1; -1])) in
    Tensor.(acc + outer)
  ) (Tensor.zeros [1; 1]) chain in
  let covariance = Tensor.(cov_sum / float_value (n_float -. 1.0)) in
  let ess = effective_sample_size chain in
  let acc_rate = List.fold_left (fun acc x ->
    if Tensor.float_value x > 0.0 then acc +. 1.0 else acc
  ) 0.0 chain /. n_float in
  {
    acceptance_rate = acc_rate;
    effective_sample_size = ess;
    mean;
    covariance;
  }

let stable_log_sum_exp xs =
  let max_x = List.fold_left max neg_infinity xs in
  let sum = List.fold_left (fun acc x ->
    acc +. exp (x -. max_x)
  ) 0. xs in
  max_x +. log sum

let numerically_stable_ratio num den =
  if abs_float den < 1e-10 then
    if abs_float num < 1e-10 then 1.0
    else if num > 0. then infinity
    else neg_infinity
  else num /. den

(* Numerical stability and precision handling *)
module NumericalStability = struct
  type numerical_params = {
    epsilon : float;
    max_condition : float;
    min_eigenvalue : float;
    max_gradient : float;
  }

  let default_params = {
    epsilon = 1e-8;
    max_condition = 1e6;
    min_eigenvalue = 1e-6;
    max_gradient = 1e3;
  }

  type stability_check = {
    condition_number : float;
    eigenvalue_range : float * float;
    gradient_norm : float;
    has_nans : bool;
  }

  let check_matrix_stability mat params =
    let eigenvalues, _ = Tensor.linalg_eigh mat in
    let min_eig = Tensor.min eigenvalues |> Tensor.float_value in
    let max_eig = Tensor.max eigenvalues |> Tensor.float_value in
    let cond = if min_eig = 0. then infinity else max_eig /. min_eig in
    cond < params.max_condition && min_eig >= params.min_eigenvalue

  let stabilize_covariance cov params =
    if check_matrix_stability cov params then cov
    else
      let d = Tensor.size cov |> List.hd in
      let eigenvalues, eigenvectors = Tensor.linalg_eigh cov in
      let stabilized_eigs = Tensor.map (fun x ->
        if x < params.min_eigenvalue then params.min_eigenvalue else x
      ) eigenvalues in
      Tensor.(matmul (matmul eigenvectors (diag stabilized_eigs))
                     (transpose eigenvectors))

  let compute_stable_gradient f x params =
    let d = Tensor.size x |> List.hd in
    let grad = Tensor.zeros [d] in
    
    for i = 0 to d - 1 do
      let h = Tensor.zeros [d] in
      Tensor.copy_ ~src:(Tensor.float_value params.epsilon |> Tensor.of_float1)
                   ~dst:(Tensor.slice h [i; i+1]);
      let fplus = f Tensor.(x + h) |> Tensor.float_value in
      let fminus = f Tensor.(x - h) |> Tensor.float_value in
      let deriv = (fplus -. fminus) /. (2. *. params.epsilon) in
      Tensor.copy_ ~src:(deriv |> Tensor.of_float1)
                   ~dst:(Tensor.slice grad [i; i+1])
    done;
    
    let norm = Tensor.norm grad ~p:2 ~dim:[0] ~keepdim:false 
               |> Tensor.float_value in
    if norm > params.max_gradient then
      Tensor.(grad * float_value (params.max_gradient /. norm))
    else grad

  let check_numerical_stability x f params =
    let grad = compute_stable_gradient f x params in
    let grad_norm = Tensor.norm grad ~p:2 ~dim:[0] ~keepdim:false 
                   |> Tensor.float_value in
    let has_nans = Tensor.isnan x |> Tensor.any |> Tensor.to_bool0_int > 0 in
    
    (* Compute local Hessian *)
    let d = Tensor.size x |> List.hd in
    let hessian = Tensor.zeros [d; d] in
    
    for i = 0 to d - 1 do
      for j = 0 to d - 1 do
        let hi = Tensor.zeros [d] in
        let hj = Tensor.zeros [d] in
        Tensor.copy_ ~src:(Tensor.float_value params.epsilon |> Tensor.of_float1)
                     ~dst:(Tensor.slice hi [i; i+1]);
        Tensor.copy_ ~src:(Tensor.float_value params.epsilon |> Tensor.of_float1)
                     ~dst:(Tensor.slice hj [j; j+1]);
        
        let fpp = f Tensor.(x + hi + hj) |> Tensor.float_value in
        let fpm = f Tensor.(x + hi - hj) |> Tensor.float_value in
        let fmp = f Tensor.(x - hi + hj) |> Tensor.float_value in
        let fmm = f Tensor.(x - hi - hj) |> Tensor.float_value in
        
        let hij = (fpp -. fpm -. fmp +. fmm) /. 
                 (4. *. params.epsilon *. params.epsilon) in
        Tensor.copy_ ~src:(hij |> Tensor.of_float1)
                     ~dst:(Tensor.slice hessian [i,j; i+1,j+1])
      done
    done;
    
    let eigenvalues, _ = Tensor.linalg_eigh hessian in
    let min_eig = Tensor.min eigenvalues |> Tensor.float_value in
    let max_eig = Tensor.max eigenvalues |> Tensor.float_value in
    let cond = if min_eig = 0. then infinity else max_eig /. min_eig in
    
    { condition_number = cond;
      eigenvalue_range = (min_eig, max_eig);
      gradient_norm = grad_norm;
      has_nans }
end

(* Complete RTF *)
module CompleteRTF = struct
  type rtf_form = 
    | RadiallySymmetric   (* R₁,₂(r) = r *)
    | UniformBounds       (* R₁,₂(r) = r(λ₂/λ₁) *)
    | UnimodalConvex      (* R₁,₂(r) = Ψ⁻¹₂(Ψ₁(r)) *)
    | GeneralForm of {    (* Custom RTF with existence checks *)
      transform : Tensor.t -> Tensor.t;
      derivative : Tensor.t -> Tensor.t;
      exists : Tensor.t -> bool;
      contour_map : Tensor.t -> float;
    }

  let check_convexity points density =
    let n = List.length points in
    let d = List.hd points |> Tensor.size |> List.hd in
    
    let is_convex = ref true in
    for i = 0 to n - 2 do
      for j = i + 1 to n - 1 do
        let x1 = List.nth points i in
        let x2 = List.nth points j in
        let alpha = Random.float 1.0 in
        let midpoint = Tensor.(x1 * float_value alpha + 
                              x2 * float_value (1. -. alpha)) in
        
        let f1 = density x1 |> Tensor.float_value in
        let f2 = density x2 |> Tensor.float_value in
        let fm = density midpoint |> Tensor.float_value in
        
        if fm > alpha *. f1 +. (1. -. alpha) *. f2 then
          is_convex := false
      done
    done;
    !is_convex

  let construct_radially_symmetric_rtf anchor_point =
    let transform r = r in
    let derivative r = Tensor.ones_like r in
    let exists _ = true in
    let contour_map r = Tensor.float_value r in
    GeneralForm { transform; derivative; exists; contour_map }

  let construct_uniform_rtf bounds =
    let lambda1, lambda2 = bounds in
    let transform r = Tensor.(r * float_value (lambda2 /. lambda1)) in
    let derivative r = Tensor.float_value (lambda2 /. lambda1) |> Tensor.ones_like in
    let exists _ = true in
    let contour_map r = Tensor.float_value r *. lambda2 /. lambda1 in
    GeneralForm { transform; derivative; exists; contour_map }

  let construct_unimodal_rtf density mode =
    let rec find_psi r theta =
      let point = Tensor.(mode + (r * theta)) in
      density point |> Tensor.float_value
    in
    
    let find_inverse_psi psi_val theta =
      let eps = 1e-6 in
      let max_iter = 100 in
      
      let rec newton_method r iter =
        if iter >= max_iter then r
        else
          let curr_psi = find_psi r theta in
          let delta_r = eps *. r in
          let next_psi = find_psi (r +. delta_r) theta in
          let derivative = (next_psi -. curr_psi) /. delta_r in
          
          if abs_float derivative < eps then r
          else
            let next_r = r -. (curr_psi -. psi_val) /. derivative in
            if abs_float (curr_psi -. psi_val) < eps then next_r
            else newton_method next_r (iter + 1)
      in
      newton_method 1.0 0
    in
    
    let transform r =
      let theta = Tensor.randn_like r in
      let normalized = Tensor.(theta / norm theta ~p:2 ~dim:[0] ~keepdim:true) in
      let psi_val = find_psi (Tensor.float_value r) normalized in
      find_inverse_psi psi_val normalized |> Tensor.of_float1
    in
    
    let derivative r =
      let eps = 1e-6 in
      let r_val = Tensor.float_value r in
      let dr = transform (Tensor.float_value (r_val +. eps) |> Tensor.of_float1) in
      let dl = transform (Tensor.float_value (r_val -. eps) |> Tensor.of_float1) in
      Tensor.((dr - dl) / float_value (2.0 *. eps))
    in
    
    let exists r =
      let theta = Tensor.randn_like r in
      let normalized = Tensor.(theta / norm theta ~p:2 ~dim:[0] ~keepdim:true) in
      let point = Tensor.(mode + (r * normalized)) in
      check_convexity [point] density
    in
    
    let contour_map r =
      let theta = Tensor.randn_like r in
      let normalized = Tensor.(theta / norm theta ~p:2 ~dim:[0] ~keepdim:true) in
      find_psi (Tensor.float_value r) normalized
    in
    
    GeneralForm { transform; derivative; exists; contour_map }

  let apply_rtf rtf_type r =
    match rtf_type with
    | RadiallySymmetric -> r
    | UniformBounds -> failwith "Requires bounds specification"
    | UnimodalConvex -> failwith "Requires density and mode specification"
    | GeneralForm { transform; _ } -> transform r

  let get_derivative rtf_type =
    match rtf_type with
    | RadiallySymmetric -> fun _ -> Tensor.ones [1]
    | UniformBounds -> failwith "Requires bounds specification"
    | UnimodalConvex -> failwith "Requires density and mode specification"
    | GeneralForm { derivative; _ } -> derivative

  let exists rtf_type point =
    match rtf_type with
    | RadiallySymmetric -> true
    | UniformBounds -> true
    | UnimodalConvex -> failwith "Requires density and mode specification"
    | GeneralForm { exists; _ } -> exists point

  let get_contour_value rtf_type point =
    match rtf_type with
    | RadiallySymmetric -> Tensor.float_value point
    | UniformBounds -> failwith "Requires bounds specification"
    | UnimodalConvex -> failwith "Requires density and mode specification"
    | GeneralForm { contour_map; _ } -> contour_map point
end

(* Coordinate system transformations *)
module CoordinateSystem = struct
  type spherical_coords = {
    r : Tensor.t;
    theta : Tensor.t array;
  }

  let cartesian_to_spherical x anchor_point =
    let diff = Tensor.(x - anchor_point) in
    let d = Tensor.size diff |> List.hd in
    let r = Tensor.norm diff ~p:2 ~dim:[0] ~keepdim:true in
    
    let rec compute_angles idx acc =
      if idx >= d - 1 then Array.of_list (List.rev acc)
      else
        let remaining_norm = Tensor.norm 
          Tensor.(slice diff [idx + 1; d]) ~p:2 ~dim:[0] ~keepdim:true in
        let theta = Tensor.atan2 remaining_norm 
          Tensor.(slice diff [idx; idx + 1]) in
        compute_angles (idx + 1) (theta :: acc)
    in
    let thetas = compute_angles 0 [] in
    { r; theta = thetas }

  let spherical_to_cartesian coords anchor_point =
    let d = Array.length coords.theta + 1 in
    let x = Tensor.zeros [d] in
    
    let rec fill_coords idx acc_sin =
      if idx >= d then x
      else if idx = d - 1 then
        let last_coord = Tensor.(coords.r * acc_sin) in
        Tensor.copy_ ~src:last_coord ~dst:(Tensor.slice x [d-1; d]);
        x
      else
        let angle = coords.theta.(idx) in
        let curr_coord = Tensor.(coords.r * acc_sin * cos angle) in
        Tensor.copy_ ~src:curr_coord ~dst:(Tensor.slice x [idx; idx+1]);
        fill_coords (idx + 1) Tensor.(acc_sin * sin angle)
    in
    let result = fill_coords 0 (Tensor.ones [1]) in
    Tensor.(result + anchor_point)

  let jacobian_spherical_to_cartesian coords =
    let d = Array.length coords.theta + 1 in
    let r = Tensor.float_value coords.r in
    let thetas = Array.map Tensor.float_value coords.theta in
    
    let rec compute_determinant n =
      if n = 1 then r
      else r ** (float (n-1)) *. sin (thetas.(n-2)) *. compute_determinant (n-1)
    in
    compute_determinant d
end

(* Transition kernel *)
module Kernel = struct
  type kernel_type =
    | Local 
    | Intrepid
    | Mixed of float  (* mixing ratio *)

  type transition_kernel = {
    name : string;
    kernel_type : kernel_type;
    generating_function : Tensor.t -> Tensor.t -> float;
    acceptance_probability : Tensor.t -> Tensor.t -> float;
    is_reversible : bool;
  }

  let make_local_kernel proposal_scale =
    let open Distribution in
    {
      name = "local";
      kernel_type = Local;
      generating_function = (fun x y ->
        let diff = Tensor.(y - x) in
        let proposal = multivariate_normal 
          ~mu:(Some x) 
          ~sigma:(Some (Tensor.eye (Tensor.size x |> List.hd) 
                       |> Tensor.mul_scalar (proposal_scale *. proposal_scale))) in
        Tensor.float_value (proposal y)
      );
      acceptance_probability = (fun x y ->
        min 1.0 (exp (Tensor.float_value Tensor.(y - x)))
      );
      is_reversible = true;
    }

  let make_intrepid_kernel config rtf =
    let open Types in
    {
      name = "intrepid";
      kernel_type = Intrepid;
      generating_function = (fun x y ->
        let rx, thetax = CoordinateSystem.cartesian_to_spherical x config.anchor_point in
        let ry, thetay = CoordinateSystem.cartesian_to_spherical y config.anchor_point in
        
        (* Angular component *)
        let angular_prob = Array.map2 (fun tx ty ->
          let diff = Tensor.float_value Tensor.(ty - tx) in
          exp (-0.5 *. diff *. diff /. (config.proposal_scale *. config.proposal_scale))
        ) thetax.theta thetay.theta |> Array.fold_left ( *. ) 1.0 in
        
        (* Radial component *)
        let radial_prob = 
          let r_ratio = Tensor.float_value Tensor.(ry.r / rx.r) in
          if r_ratio > config.radial_gamma || r_ratio < (1.0 /. config.radial_gamma) then
            0.0
          else
            1.0 /. (config.radial_gamma -. 1.0 /. config.radial_gamma) in
        
        angular_prob *. radial_prob
      );
      acceptance_probability = (fun x y ->
        let rx, _ = CoordinateSystem.cartesian_to_spherical x config.anchor_point in
        let ry, _ = CoordinateSystem.cartesian_to_spherical y config.anchor_point in
        let rtf_deriv = CompleteRTF.get_derivative rtf ry.r 
                       |> Tensor.float_value in
        min 1.0 (rtf_deriv *. exp (Tensor.float_value Tensor.(y - x)))
      );
      is_reversible = true;
    }

  let compose_kernels k1 k2 mixing_ratio =
    {
      name = "mixed";
      kernel_type = Mixed mixing_ratio;
      generating_function = (fun x y ->
        mixing_ratio *. k1.generating_function x y +.
        (1.0 -. mixing_ratio) *. k2.generating_function x y
      );
      acceptance_probability = (fun x y ->
        let alpha1 = k1.acceptance_probability x y in
        let alpha2 = k2.acceptance_probability x y in
        mixing_ratio *. alpha1 +. (1.0 -. mixing_ratio) *. alpha2
      );
      is_reversible = k1.is_reversible && k2.is_reversible;
    }
end

(* Mode finding and chain mixing *)
module ModeFinding = struct
  type mode = {
    location : Tensor.t;
    density : float;
    covariance : Tensor.t;
    weight : float;
  }

  type mixing_info = {
    between_mode_transitions : int;
    mode_visits : int array;
    last_mode : int;
    total_steps : int;
  }

  let estimate_local_mode point log_density =
    let d = Tensor.size point |> List.hd in
    let step_size = 0.01 in
    let max_iter = 100 in
    
    let rec gradient_ascent x iter =
      if iter >= max_iter then x
      else
        let grad = NumericalStability.compute_stable_gradient log_density x 
                    NumericalStability.default_params in
        let norm = Tensor.norm grad ~p:2 ~dim:[0] ~keepdim:true in
        if Tensor.float_value norm < 1e-6 then x
        else
          let next_x = Tensor.(x + (grad * float_value step_size)) in
          if Tensor.float_value (log_density next_x) > 
             Tensor.float_value (log_density x) then
            gradient_ascent next_x (iter + 1)
          else
            gradient_ascent x max_iter
    in
    gradient_ascent point 0

  let estimate_mode_covariance points mode_loc =
    let n = List.length points in
    let d = Tensor.size mode_loc |> List.hd in
    if n < d + 1 then
      Tensor.eye d
    else
      let sum_sq = List.fold_left (fun acc x ->
        let diff = Tensor.(x - mode_loc) in
        Tensor.(acc + matmul (view diff ~size:[-1; 1]) 
                            (view diff ~size:[1; -1]))
      ) (Tensor.zeros [d; d]) points in
      Tensor.(sum_sq / float_value (float (n - 1)))

  let identify_modes chain log_density =
    let n = List.length chain in
    let modes = ref [] in
    let mode_radius = 2.0 in
    
    List.iter (fun x ->
      let local_mode = estimate_local_mode x log_density in
      let new_mode = ref true in
      List.iter (fun mode ->
        let dist = Tensor.norm Tensor.(local_mode - mode.location) 
                  ~p:2 ~dim:[0] ~keepdim:false 
                  |> Tensor.float_value in
        if dist < mode_radius then
          new_mode := false
      ) !modes;

      if !new_mode then
        let nearby_points = List.filter (fun y ->
          let dist = Tensor.norm Tensor.(y - local_mode) 
                    ~p:2 ~dim:[0] ~keepdim:false 
                    |> Tensor.float_value in
          dist < mode_radius
        ) chain in
        let cov = estimate_mode_covariance nearby_points local_mode in
        modes := {
          location = local_mode;
          density = Tensor.float_value (log_density local_mode);
          covariance = cov;
          weight = float (List.length nearby_points) /. float n;
        } :: !modes
    ) chain;
    List.rev !modes

  let track_mixing chain modes =
    let n = List.length chain in
    let n_modes = List.length modes in
    let mode_visits = Array.make n_modes 0 in
    let between_transitions = ref 0 in
    let last_mode = ref (-1) in
    
    let find_mode x =
      let best_mode = ref (-1) in
      let min_dist = ref infinity in
      List.iteri (fun i mode ->
        let dist = Tensor.norm Tensor.(x - mode.location) 
                  ~p:2 ~dim:[0] ~keepdim:false 
                  |> Tensor.float_value in
        if dist < !min_dist then (
          min_dist := dist;
          best_mode := i
        )
      ) modes;
      !best_mode
    in
    
    List.iter (fun x ->
      let current_mode = find_mode x in
      if current_mode >= 0 then (
        mode_visits.(current_mode) <- mode_visits.(current_mode) + 1;
        if !last_mode >= 0 && current_mode <> !last_mode then
          incr between_transitions;
        last_mode := current_mode
      )
    ) chain;
    
    {
      between_mode_transitions = !between_transitions;
      mode_visits;
      last_mode = !last_mode;
      total_steps = n;
    }
end

(* Chain analysis and monitoring *)
module ChainAnalysis = struct
  type analysis_config = {
    window_size : int;
    min_samples : int;
    convergence_threshold : float;
    stability_threshold : float;
  }

  type chain_metrics = {
    effective_samples : float;
    acceptance_rate : float;
    exploration_score : float;
    stability_score : float;
    mode_coverage : float;
  }

  type chain_state = {
    metrics : chain_metrics;
    modes : ModeFinding.mode list;
    transitions : int;
    stable_windows : int;
  }

  let compute_chain_metrics chain config =
    let n = List.length chain in
    let windows = (n + config.window_size - 1) / config.window_size in
    
    let window_stats = Array.init windows (fun i ->
      let start = i * config.window_size in
      let length = min config.window_size (n - start) in
      let window_chain = List.filteri (fun j _ -> 
        j >= start && j < start + length) chain in
      compute_stats window_chain
    ) in
    
    (* Compute effective sample size *)
    let total_var = Array.fold_left (fun acc stats ->
      acc +. Tensor.float_value stats.covariance
    ) 0.0 window_stats /. float windows in
    
    let within_var = Array.fold_left (fun acc stats ->
      let diff = Tensor.(stats.mean - window_stats.(0).mean) in
      acc +. Tensor.float_value Tensor.(diff * diff)
    ) 0.0 window_stats /. float (windows - 1) in
    
    let ess = float n *. total_var /. (total_var +. within_var) in
    
    (* Compute mode coverage *)
    let modes = ModeFinding.identify_modes chain (fun x -> x) in
    let mixing = ModeFinding.track_mixing chain modes in
    let coverage = float (List.length modes |> fun n -> 
      if n = 0 then 1 else n) in
    
    (* Compute stability score *)
    let stability = float mixing.between_mode_transitions /. 
                   float mixing.total_steps in
    
    {
      effective_samples = ess;
      acceptance_rate = float mixing.between_mode_transitions /. 
                       float mixing.total_steps;
      exploration_score = coverage;
      stability_score = stability;
      mode_coverage = coverage
    }

  let monitor_chain_progress chain config =
    let metrics = compute_chain_metrics chain config in
    let modes = ModeFinding.identify_modes chain (fun x -> x) in
    let mixing = ModeFinding.track_mixing chain modes in
    
    {
      metrics;
      modes;
      transitions = mixing.between_mode_transitions;
      stable_windows = if metrics.stability_score > config.stability_threshold 
                      then 1 else 0
    }
end

(* Parameter space exploration *)
module ParameterExploration = struct
  type region_type =
    | Known       (* Already explored *)
    | Boundary    (* Edge of explored space *)
    | Unknown     (* Not yet explored *)

  type exploration_stats = {
    visited_regions : (int * int) list;
    boundary_points : Tensor.t list;
    unknown_directions : Tensor.t list;
    exploration_score : float;
  }

  let grid_size = 20
  let boundary_threshold = 0.1

  let get_grid_coords point bounds =
    let lower, upper = bounds in
    let dims = Tensor.size point |> List.hd in
    let normalized = Tensor.((point - lower) / (upper - lower)) in
    Array.init dims (fun i ->
      let coord = Tensor.(get normalized [i]) |> Tensor.float_value in
      int_of_float (coord *. float grid_size)
    )

  let classify_region points bounds point =
    let coords = get_grid_coords point bounds in
    let nearby = List.filter (fun p ->
      let p_coords = get_grid_coords p bounds in
      Array.for_all2 (fun c1 c2 -> abs (c1 - c2) <= 1) coords p_coords
    ) points in
    
    if List.length nearby > 0 then
      let distances = List.map (fun p ->
        Tensor.norm Tensor.(point - p) ~p:2 ~dim:[0] ~keepdim:false 
        |> Tensor.float_value
      ) nearby in
      if List.exists (fun d -> d <= boundary_threshold) distances then
        Known
      else
        Boundary
    else
      Unknown

  let find_exploration_direction points bounds current =
    let dims = Tensor.size current |> List.hd in
    let best_dir = ref None in
    let max_emptiness = ref 0.0 in
    
    for _ = 1 to 100 do
      let dir = Tensor.randn [dims] in
      let normalized = Tensor.(dir / norm dir ~p:2 ~dim:[0] ~keepdim:true) in
      
      let emptiness = ref 0.0 in
      for i = 1 to 10 do
        let probe = Tensor.(current + (normalized * float_value (float i *. 0.1))) in
        match classify_region points bounds probe with
        | Unknown -> emptiness := !emptiness +. 1.0
        | Boundary -> emptiness := !emptiness +. 0.5
        | Known -> ()
      done;
      
      if !emptiness > !max_emptiness then (
        max_emptiness := !emptiness;
        best_dir := Some normalized
      )
    done;
    !best_dir

  let explore_region points bounds current =
    match classify_region points bounds current with
    | Known -> None
    | Boundary ->
        find_exploration_direction points bounds current
    | Unknown ->
        Some Tensor.(neg current / norm current ~p:2 ~dim:[0] ~keepdim:true)

  let compute_exploration_score points bounds =
    let total_regions = grid_size * grid_size in
    let visited = Hashtbl.create total_regions in
    
    List.iter (fun p ->
      let coords = get_grid_coords p bounds in
      let key = (Array.get coords 0, Array.get coords 1) in
      Hashtbl.replace visited key true
    ) points;
    
    float (Hashtbl.length visited) /. float total_regions
end

(* Chain timing and adaptation *)
module ChainTiming = struct
  type timing_params = {
    burn_in : int;
    min_adaptation_window : int;
    max_adaptation_window : int;
    stabilization_window : int;
    convergence_check_interval : int;
  }

  type adaptation_phase =
    | BurnIn
    | Adaptation
    | Stationary
    | Converged

  type timing_stats = {
    phase : adaptation_phase;
    current_window : int;
    elapsed_steps : int;
    stable_windows : int;
  }

  let determine_phase stats params =
    if stats.elapsed_steps < params.burn_in then
      BurnIn
    else if stats.stable_windows < 3 then
      Adaptation
    else if stats.current_window < params.max_adaptation_window then
      Stationary
    else
      Converged

  let adapt_window_size stats params =
    let current = stats.current_window in
    if stats.phase = Adaptation then
      min (current * 2) params.max_adaptation_window
    else if stats.phase = Stationary then
      min (current + params.stabilization_window) params.max_adaptation_window
    else
      current
end

(** Main MCMC *)
module IntrepidMCMC = struct
  type mcmc_config = {
    n_chains : int;
    n_samples : int;
    exploration_ratio : float;
    numerical_params : numerical_params;
    analysis_config : ChainAnalysis.analysis_config;
    timing_params : ChainTiming.timing_params;
    convergence_criteria : ChainConvergence.convergence_criteria;
  }

  type run_stats = {
    chain_states : ChainAnalysis.chain_state array;
    convergence : bool;
    total_modes : int;
    execution_time : float;
  }

  let default_timing_params = {
    ChainTiming.burn_in = 1000;
    min_adaptation_window = 100;
    max_adaptation_window = 5000;
    stabilization_window = 500;
    convergence_check_interval = 100;
  }

  let create_default_config n_dims = {
    n_chains = 4;
    n_samples = 10000;
    exploration_ratio = 0.2;
    numerical_params = default_params;
    analysis_config = {
      window_size = 100;
      min_samples = 1000;
      convergence_threshold = 0.01;
      stability_threshold = 0.95;
    };
    timing_params = default_timing_params;
    convergence_criteria = {
      ChainConvergence.psrf_threshold = 1.1;
      ess_min = 100.0;
      geweke_zscore = 1.96;
      chain_corr = 0.1;
    };
  }

  let run_single_chain config initial_point parent_pdf target_pdf =
    let kernel = Kernel.make_intrepid_kernel 
      { exploration_ratio = config.exploration_ratio;
        n_dims = Tensor.size initial_point |> List.hd;
        anchor_point = Tensor.zeros [Tensor.size initial_point |> List.hd];
        proposal_scale = 0.1;
        angular_scales = Array.make 
          ((Tensor.size initial_point |> List.hd) - 1) (Float.pi /. 4.0);
        radial_gamma = 2.0;
        min_accept_rate = 0.234;
        max_accept_rate = 0.44;
      } 
      (CompleteRTF.RadiallySymmetric) in

    let rec sample_chain chain timing_stats n =
      if n <= 0 then List.rev chain, timing_stats
      else
        let current = List.hd chain in
        let phase = ChainTiming.determine_phase timing_stats config.timing_params in
        let window_size = ChainTiming.adapt_window_size timing_stats 
                           config.timing_params in

        let proposal = if phase = ChainTiming.BurnIn then
          Tensor.(current + (randn_like current * float_value 0.1))
        else
          let proposed = 
            if Random.float 1.0 < config.exploration_ratio then
              let _ = kernel.Kernel.generating_function current in
              let rx, thetax = CoordinateSystem.cartesian_to_spherical 
                                current kernel.Kernel.name in
              let ry = CompleteRTF.apply_rtf 
                        CompleteRTF.RadiallySymmetric rx.r in
              let normalized = Tensor.(randn [Tensor.size current |> List.hd]) in
              let thetay = Tensor.(normalized / norm normalized 
                                  ~p:2 ~dim:[0] ~keepdim:true) in
              CoordinateSystem.spherical_to_cartesian 
                { r = ry; theta = [|thetay|] } 
                (Tensor.zeros [Tensor.size current |> List.hd])
            else
              Tensor.(current + (randn_like current * float_value 0.1))
          in
          proposed
        in

        let current_log_prob = target_pdf current |> Tensor.float_value in
        let proposal_log_prob = target_pdf proposal |> Tensor.float_value in
        let accept_prob = exp (proposal_log_prob -. current_log_prob) in
        let accepted = Random.float 1.0 < accept_prob in

        let next_point = if accepted then proposal else current in
        let new_timing_stats = {
          timing_stats with
          elapsed_steps = timing_stats.elapsed_steps + 1;
          current_window = window_size;
          stable_windows = 
            if accepted && phase = ChainTiming.Stationary then
              timing_stats.stable_windows + 1
            else
              timing_stats.stable_windows;
        } in

        sample_chain (next_point :: chain) new_timing_stats (n - 1)
    in

    let initial_timing_stats = {
      ChainTiming.phase = BurnIn;
      current_window = config.timing_params.min_adaptation_window;
      elapsed_steps = 0;
      stable_windows = 0;
    } in

    sample_chain [initial_point] initial_timing_stats config.n_samples

  let run_parallel config initial_points parent_pdf target_pdf =
    let start_time = Unix.gettimeofday () in
    
    let run_chain idx init_point =
      Printf.printf "Starting chain %d\n" idx;
      run_single_chain config init_point parent_pdf target_pdf
    in
    
    let chains_and_stats = Array.mapi run_chain initial_points in
    let chains = Array.map fst chains_and_stats in
    
    let final_states = Array.map (fun chain ->
      ChainAnalysis.monitor_chain_progress chain config.analysis_config
    ) chains in
    
    let all_chains = Array.to_list chains in
    let converged = ChainConvergence.check_convergence 
                     all_chains config.convergence_criteria in
    
    let all_modes = Array.fold_left (fun acc state ->
      List.fold_left (fun acc mode ->
        if not (List.exists (fun m ->
          let dist = Tensor.norm 
            Tensor.(mode.ModeFinding.location - m.ModeFinding.location)
            ~p:2 ~dim:[0] ~keepdim:false 
            |> Tensor.float_value in
          dist < 1e-6
        ) acc) then
          mode :: acc
        else acc
      ) acc state.ChainAnalysis.modes
    ) [] final_states in
    
    {
      chain_states = final_states;
      convergence = converged;
      total_modes = List.length all_modes;
      execution_time = Unix.gettimeofday () -. start_time;
    }
end