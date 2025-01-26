open Torch

(** Function with gradient *)
type function_with_gradient = {
    f: Tensor.t -> Tensor.t;  
    grad_f: Tensor.t -> Tensor.t;
}

(** Active subspace *)
type active_subspace = {
    dimension: int;
    eigenvectors: Tensor.t;
    eigenvalues: Tensor.t;
}

(** Sample result containing points and values *)
type sample_result = {
    points: Tensor.t;
    values: Tensor.t;
    gradients: Tensor.t;
}

(** Measure type *)
type measure = {
    density: Tensor.t -> float;
    support: Tensor.t -> bool;
    dimension: int;
}

(** Conditional measure *)
type conditional_measure = {
    base_measure: measure;
    condition: Tensor.t -> bool;
    conditional_density: Tensor.t -> Tensor.t -> float;
}

(** Level configuration for multilevel method *)
type level_config = {
    samples: int;
    dimension: int;
    poly_degree: int;
}

(** Result of multilevel computation *)
type mlas_result = {
    active_subspaces: active_subspace list;
    level_approximations: Tensor.t list;
    total_approximation: Tensor.t;
}

(** Borel measures *)
module BorelMeasure = struct
    let measure mu set =
      mu.density set

    let null_set set =
      Tensor.sum set |> Tensor.to_float0_exn = 0.

    let absolutely_continuous mu nu set =
      if nu.density set = 0. then mu.density set = 0.
      else true

    let radon_nikodym_derivative mu nu x =
      if not (absolutely_continuous mu nu set) then
        invalid_arg "Measure mu is not absolutely continuous w.r.t. nu";
      mu.density x /. nu.density x
end

(** Lebesgue integration *)
module LebesgueIntegral = struct
    let monte_carlo_integrate f measure n_samples =
      let points = Tensor.randn [n_samples; measure.dimension] in
      let values = List.init n_samples (fun i ->
        let x = Tensor.select points ~dim:0 ~index:i in
        f x *. measure.density x) in
      List.fold_left (+.) 0. values /. float_of_int n_samples

    let adaptive_integrate f measure tol =
      let rec integrate_interval a b depth acc =
        let mid = (a +. b) /. 2. in
        let fa = f (Tensor.of_float1 [|a|]) in
        let fm = f (Tensor.of_float1 [|mid|]) in
        let fb = f (Tensor.of_float1 [|b|]) in
        
        let area1 = (fa +. fb) *. (b -. a) /. 2. in
        let area2 = (fa +. 4. *. fm +. fb) *. (b -. a) /. 6. in
        
        if depth > 50 || abs_float (area1 -. area2) < tol then
          area2
        else
          let left = integrate_interval a mid (depth + 1) (acc /. 2.) in
          let right = integrate_interval mid b (depth + 1) (acc /. 2.) in
          left +. right in
      
      integrate_interval (-5.) 5. 0 1.
end

(** Tensor product measure *)
module TensorProduct = struct
    let create mu1 mu2 =
      let product_density x =
        let d1 = mu1.dimension in
        let x1 = Tensor.narrow x ~dim:0 ~start:0 ~length:d1 in
        let x2 = Tensor.narrow x ~dim:0 ~start:d1 ~length:mu2.dimension in
        mu1.density x1 *. mu2.density x2 in
      
      {density = product_density;
       support = (fun x -> 
         let d1 = mu1.dimension in
         let x1 = Tensor.narrow x ~dim:0 ~start:0 ~length:d1 in
         let x2 = Tensor.narrow x ~dim:0 ~start:d1 ~length:mu2.dimension in
         mu1.support x1 && mu2.support x2);
       dimension = mu1.dimension + mu2.dimension}

    let tensor_product_list measures =
      List.fold_left (fun acc m -> create acc m) 
        (List.hd measures) (List.tl measures)
end

(** Core Active Subspace *)
module ActiveSubspace = struct
  (** Helper to compute outer product of gradient samples *)
  let compute_gradient_outer_product grads = 
    let n = Tensor.size grads 0 in
    let d = Tensor.size grads 1 in
    let grads_reshaped = Tensor.reshape grads [n; d; 1] in
    let grads_t = Tensor.reshape grads [n; 1; d] in
    let products = Tensor.matmul grads_t grads_reshaped in
    Tensor.mean products ~dim:[0] ~keepdim:false

  (** Draw random samples for Monte Carlo estimation *)
  let draw_samples n d distribution =
    match distribution with
    | `Gaussian -> Tensor.randn [n; d]
    | `Uniform -> Tensor.sub (Tensor.mul (Tensor.rand [n; d]) 
                              (Tensor.of_float 2.)) 
                            (Tensor.of_float 1.)

  (** Compute truncated SVD *)
  let truncated_svd matrix r =
    let u, s, v = Tensor.svd matrix in
    let u_r = Tensor.narrow u ~dim:1 ~start:0 ~length:r in
    let s_r = Tensor.narrow s ~dim:0 ~start:0 ~length:r in
    let v_r = Tensor.narrow v ~dim:1 ~start:0 ~length:r in
    (u_r, s_r, v_r)

  (** Main Active Subspace estimation routine *)
  let estimate_active_subspace func n_samples dimension =
    let points = draw_samples n_samples dimension `Gaussian in
    let gradients = Tensor.stack 
      (List.init n_samples (fun i -> 
        func.grad_f (Tensor.select points ~dim:0 ~index:i)))
      ~dim:0 in
    let cov = compute_gradient_outer_product gradients in
    let u_r, s_r, v_r = truncated_svd cov dimension in
    
    {dimension;
     eigenvectors = u_r;
     eigenvalues = s_r}

  (** Project points onto active subspace *)
  let project_points points active_subspace =
    Tensor.matmul points active_subspace.eigenvectors
end

(** Polynomial basis and approximation *)
module PolynomialBasis = struct
  type multi_index = int array
  
  type polynomial_space = {
    dimension: int;
    max_degree: int;
    active_vars: int array;
    inactive_vars: int array;
  }

  (** Hermite polynomial *)
  module HermiteBasis = struct
    let hermite_recursion_coeffs n =
      let a_n = if n = 0 then 0. else sqrt(float_of_int n) in
      let b_n = 0. in
      (a_n, b_n)

    let rec evaluate_hermite n x =
      match n with
      | 0 -> Tensor.ones_like x
      | 1 -> x
      | n ->
          let h_prev = evaluate_hermite (n-2) x in
          let h_curr = evaluate_hermite (n-1) x in
          let (a_n, b_n) = hermite_recursion_coeffs (n-1) in
          let term1 = Tensor.mul_scalar x (sqrt (2. /. float_of_int n)) in
          let term2 = Tensor.mul_scalar h_curr (b_n /. float_of_int n) in
          let term3 = Tensor.mul_scalar h_prev 
            (a_n *. sqrt (float_of_int (n-1)) /. float_of_int n) in
          Tensor.sub (Tensor.sub term1 term2) term3

    let evaluate_multivariate index points =
      let n = Tensor.size points 0 in
      let d = Tensor.size points 1 in
      assert (Array.length index = d);
      
      let univariate_polys = Array.mapi (fun i deg ->
        let x_i = Tensor.select points ~dim:1 ~index:i in
        evaluate_hermite deg x_i
      ) index in
      
      Array.fold_left (fun acc poly ->
        Tensor.mul acc poly
      ) (Tensor.ones [n]) univariate_polys

    let generate_normalized_basis space points =
      let total_dim = space.dimension in
      let active_dim = Array.length space.active_vars in
      
      let rec generate_indices curr_degree acc =
        if curr_degree > space.max_degree then acc
        else
          let new_indices = Array.make active_dim 0 in
          let rec fill_indices pos sum =
            if pos = active_dim then
              if sum = curr_degree then [Array.copy new_indices] else []
            else
              let max_val = min (curr_degree - sum) curr_degree in
              let rec try_values v acc =
                if v > max_val then acc
                else begin
                  new_indices.(pos) <- v;
                  let next_acc = acc @ fill_indices (pos + 1) (sum + v) in
                  new_indices.(pos) <- 0;
                  try_values (v + 1) next_acc
                end in
              try_values 0 [] in
          let degree_indices = fill_indices 0 0 in
          generate_indices (curr_degree + 1) (acc @ degree_indices) in
      
      let active_indices = generate_indices 0 [] in
      
      let extend_index idx =
        let full_idx = Array.make total_dim 0 in
        Array.iteri (fun i v -> 
          full_idx.(space.active_vars.(i)) <- v) idx;
        full_idx in
      
      List.map (fun idx -> 
        let full_idx = extend_index idx in
        evaluate_multivariate full_idx points) active_indices
  end

  (** Tensor product polynomial space *)
  module TensorProduct = struct
    let create_space active_vars inactive_vars max_degree = {
      dimension = Array.length active_vars + Array.length inactive_vars;
      max_degree;
      active_vars;
      inactive_vars;
    }

    let generate_basis space points =
      HermiteBasis.generate_normalized_basis space points

    let project_function f space points measure =
      let basis = generate_basis space points in
      let n_basis = List.length basis in
      
      let gram = Tensor.zeros [n_basis; n_basis] in
      List.iteri (fun i phi_i ->
        List.iteri (fun j phi_j ->
          let prod = LebesgueIntegral.monte_carlo_integrate
            (fun x -> Tensor.mul phi_i phi_j) measure 1000 in
          Tensor.set gram [i; j] prod) basis) basis;
      
      let rhs = List.map (fun phi ->
        LebesgueIntegral.monte_carlo_integrate
          (fun x -> Tensor.mul (f x) phi) measure 1000) basis in
      let rhs = Tensor.of_float1 (Array.of_list rhs) in
      
      let coeffs = Tensor.lstsq gram rhs in
      
      fun x ->
        let basis_vals = List.map (fun phi -> phi x) basis in
        let basis_tensor = Tensor.stack basis_vals ~dim:1 in
        Tensor.matmul basis_tensor coeffs
  end

  (** Adaptive polynomial basis selection *)
  module AdaptiveBasis = struct
    type adaptivity_criterion = 
      | LegendreCoeffDecay
      | HermiteCoeffDecay
      | SobolevNorm

    let estimate_coeff_decay basis_vals coeffs =
      let sorted_coeffs = 
        Tensor.sort (Tensor.abs coeffs) ~descending:true in
      let n = Tensor.size sorted_coeffs 0 in
      let log_coeffs = Tensor.log sorted_coeffs in
      let log_indices = 
        Tensor.log (Tensor.arange ~start:1. 
                     ~end_:(float_of_int n +. 1.) ~step:1.) in
      
      let x = log_indices in
      let y = log_coeffs in
      let n_float = float_of_int n in
      let sum_x = Tensor.sum x |> Tensor.to_float0_exn in
      let sum_y = Tensor.sum y |> Tensor.to_float0_exn in
      let sum_xy = Tensor.sum (Tensor.mul x y) |> Tensor.to_float0_exn in
      let sum_xx = Tensor.sum (Tensor.mul x x) |> Tensor.to_float0_exn in
      
      (n_float *. sum_xy -. sum_x *. sum_y) /. 
      (n_float *. sum_xx -. sum_x *. sum_x)

    let select_optimal_degree f space points measure criterion tol =
      let max_degree = space.max_degree in
      let curr_degree = ref 1 in
      let best_error = ref infinity in
      let best_degree = ref 1 in
      
      while !curr_degree <= max_degree do
        let test_space = {space with max_degree = !curr_degree} in
        let basis = TensorProduct.generate_basis test_space points in
        let proj = TensorProduct.project_function f test_space points measure in
        
        let error = match criterion with
          | LegendreCoeffDecay 
          | HermiteCoeffDecay ->
              let basis_vals = List.map (fun phi -> phi points) basis in
              let basis_tensor = Tensor.stack basis_vals ~dim:1 in
              let coeffs = Tensor.lstsq basis_tensor (f points) in
              -.(estimate_coeff_decay basis_vals coeffs)
          | SobolevNorm ->
              let diff = Tensor.sub (f points) (proj points) in
              Tensor.norm diff ~p:2 |> Tensor.to_float0_exn in
        
        if error < !best_error then begin
          best_error := error;
          best_degree := !curr_degree
        end;
        
        if error < tol then
          curr_degree := max_degree + 1
        else
          incr curr_degree
      done;
      
      !best_degree
  end
end

(** Error analysis and optimization *)
module ErrorAnalysis = struct
  (** SVD error analysis *)
  module SVDError = struct
    type svd_error = {
      truncation_error: float;
      reconstruction_error: float;
      condition_number: float;
      rank_selection_error: float;
    }

    let compute_truncation_error s r =
      let total_energy = Tensor.sum s |> Tensor.to_float0_exn in
      let truncated_energy = 
        Tensor.sum (Tensor.narrow s ~dim:0 ~start:r 
                     ~length:(Tensor.size s 0 - r)) 
        |> Tensor.to_float0_exn in
      truncated_energy /. total_energy

    let analyze_reconstruction matrix u s v r =
      let reconstructed = 
        Tensor.matmul 
          (Tensor.matmul 
             (Tensor.narrow u ~dim:1 ~start:0 ~length:r)
             (Tensor.diag (Tensor.narrow s ~dim:0 ~start:0 ~length:r)))
          (Tensor.transpose 
             (Tensor.narrow v ~dim:1 ~start:0 ~length:r) 
             ~dim0:0 ~dim1:1) in
      let error = Tensor.sub matrix reconstructed in
      Tensor.norm error ~p:2 |> Tensor.to_float0_exn

    let analyze_svd matrix r =
      let u, s, v = Tensor.svd matrix in
      let trunc_error = compute_truncation_error s r in
      let recon_error = analyze_reconstruction matrix u s v r in
      let cond_num = 
        (Tensor.get s [0] |> Tensor.to_float0_exn) /.
        (Tensor.get s [Tensor.size s 0 - 1] |> Tensor.to_float0_exn) in
      let opt_rank = select_optimal_rank s 1e-6 in
      let rank_error = abs_float (float_of_int (opt_rank - r)) /.
                      float_of_int opt_rank in
      
      {truncation_error = trunc_error;
       reconstruction_error = recon_error;
       condition_number = cond_num;
       rank_selection_error = rank_error}
  end

  (** Hierarchical error estimation *)
  module HierarchicalError = struct
    type error_decomposition = {
      level_errors: float array;
      total_error: float;
      relative_contributions: float array;
    }

    let compute_level_errors fine_approx coarse_approx points levels =
      Array.init levels (fun l ->
        let level_points = 
          Tensor.narrow points ~dim:0 ~start:0 
            ~length:((Tensor.size points 0) / (l + 1)) in
        let fine_vals = fine_approx level_points in
        let coarse_vals = coarse_approx level_points in
        let diff = Tensor.sub fine_vals coarse_vals in
        Tensor.norm diff ~p:2 |> Tensor.to_float0_exn)

    let analyze_error_contributions errors =
      let total = Array.fold_left (+.) 0. errors in
      let relative = Array.map (fun e -> e /. total) errors in
      {level_errors = errors;
       total_error = total;
       relative_contributions = relative}

    let estimate_convergence_rates errors levels =
      Array.init (levels - 1) (fun i ->
        log (errors.(i) /. errors.(i + 1)) /.
        log (float_of_int (i + 2) /. float_of_int (i + 1)))
  end
end

(** Multilevel Active Subspaces *)
module MultilevelAS = struct
  type mlas_config = {
    max_levels: int;
    initial_rank: int;
    max_rank: int;
    tol: float;
    initial_samples: int;
  }

  (** Level management *)
  module LevelManager = struct
    type level_data = {
      active_subspace: active_subspace;
      polynomial_approx: PolynomialBasis.polynomial_space;
      error_estimate: float;
      work_estimate: float;
    }

    let create_level func n_samples rank poly_degree =
      let as_result = estimate_active_subspace func n_samples rank in
      let points = draw_samples n_samples rank `Gaussian in
      
      let poly_space = PolynomialBasis.TensorProduct.create_space
        [|0|] [||] poly_degree in
      
      {active_subspace = as_result;
       polynomial_approx = poly_space;
       error_estimate = 0.0;  
       work_estimate = float_of_int (n_samples * rank)}

    let update_error_estimates levels points =
      Array.mapi (fun i level ->
        let next_level = 
          if i + 1 < Array.length levels 
          then Some levels.(i + 1) 
          else None in
        
        match next_level with
        | None -> level
        | Some next ->
            let error = HierarchicalError.compute_level_errors
              (fun x -> Tensor.matmul x level.active_subspace.eigenvectors)
              (fun x -> Tensor.matmul x next.active_subspace.eigenvectors)
              points 1 in
            {level with error_estimate = error.(0)}) levels
  end

  (** Work balancing *)
  module WorkBalance = struct
    let optimize_work_distribution levels target_error =
      let total_work = Array.fold_left 
        (fun acc l -> acc +. l.LevelManager.work_estimate) 0. levels in
      
      let optimal_distribution = Array.map (fun level ->
        let ratio = sqrt (level.LevelManager.error_estimate /. 
                         level.LevelManager.work_estimate) in
        let new_work = total_work *. ratio /. 
          (Array.fold_left (fun acc l ->
             acc +. sqrt (l.LevelManager.error_estimate /. 
                         l.LevelManager.work_estimate)) 0. levels) in
        {level with LevelManager.work_estimate = new_work}) levels in
      
      optimal_distribution
  end

  (** MLAS *)
  let run_mlas config func points =
    let levels = ref [||] in
    
    (* Initialize first level *)
    levels := Array.make 1 
      (LevelManager.create_level func 
         config.initial_samples 
         config.initial_rank 2);
    
    let continue = ref true in
    while !continue && Array.length !levels < config.max_levels do
      (* Update error estimates *)
      levels := LevelManager.update_error_estimates !levels points;
      
      (* Check convergence *)
      let total_error = Array.fold_left 
        (fun acc l -> acc +. l.LevelManager.error_estimate) 0. !levels in
      
      if total_error < config.tol then
        continue := false
      else begin
        (* Add new level *)
        let new_level = LevelManager.create_level func
          (config.initial_samples * Array.length !levels)
          (min (config.initial_rank * Array.length !levels) 
             config.max_rank)
          (2 + Array.length !levels) in
        
        levels := Array.append !levels [|new_level|];
        
        (* Optimize work distribution *)
        let optimal_dist = 
          WorkBalance.optimize_work_distribution !levels total_error in
        levels := optimal_dist
      end
    done;
    
    (* Construct final approximation *)
    let approx = Array.fold_left (fun acc level ->
      let level_contrib = Tensor.matmul points 
        level.LevelManager.active_subspace.eigenvectors in
      match acc with
      | None -> Some level_contrib
      | Some prev -> Some (Tensor.add prev level_contrib)
    ) None !levels in
    
    {active_subspaces = Array.to_list 
       (Array.map (fun l -> l.LevelManager.active_subspace) !levels);
     level_approximations = Array.to_list 
       (Array.map (fun l -> 
          Tensor.matmul points l.LevelManager.active_subspace.eigenvectors) 
          !levels);
     total_approximation = Option.get approx}
end

(** Numerical utilities and stability handling *)
module Numerics = struct
    let chunk_operation tensor chunk_size op =
      let n = Tensor.size tensor 0 in
      let num_chunks = (n + chunk_size - 1) / chunk_size in
      let result = Tensor.zeros_like tensor in
      
      for i = 0 to num_chunks - 1 do
        let start_idx = i * chunk_size in
        let end_idx = min (start_idx + chunk_size) n in
        let chunk = Tensor.narrow tensor ~dim:0 
          ~start:start_idx ~length:(end_idx - start_idx) in
        let processed = op chunk in
        Tensor.copy_ 
          (Tensor.narrow result ~dim:0 
             ~start:start_idx ~length:(end_idx - start_idx))
          processed
      done;
      result

  (** Stability monitoring *)
  module StabilityMonitor = struct
    type stability_status = {
      condition_number: float;
      residual_norm: float;
      gradient_variation: float;
      is_stable: bool;
    }

    let check_stability matrix gradient =
      let u, s, v = Tensor.svd matrix in
      let s_array = Tensor.to_float1 s in
      let cond_num = s_array.(0) /. 
        s_array.(Array.length s_array - 1) in
      
      let grad_norm = Tensor.norm gradient ~p:2 
        |> Tensor.to_float0_exn in
      let grad_var = Tensor.std gradient ~dim:[0] ~unbiased:true
        |> Tensor.mean
        |> Tensor.to_float0_exn in
      
      {condition_number = cond_num;
       residual_norm = grad_norm;
       gradient_variation = grad_var;
       is_stable = cond_num < 1e6 && grad_var < 1e3}

    let stabilize_computation status matrix gradient =
      if not status.is_stable then
        (* Add regularization *)
        let identity = Tensor.eye (Tensor.size matrix 0) in
        let reg_factor = 1e-6 *. 
          Tensor.trace matrix |> Tensor.to_float0_exn in
        (Tensor.add matrix (Tensor.mul_scalar identity reg_factor), true)
      else
        (matrix, false)
  end
end