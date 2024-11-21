open Torch
open Spaces

module Linear = struct
  type belief_structure = {
    mean_x: Tensor.t;
    mean_d: Tensor.t;
    var_x: Tensor.t;
    var_d: Tensor.t;
    cov_xd: Tensor.t;
    inner_product_space: LinearAlgebra.InnerProduct.t;
  }

  let validate_dimensions belief =
    let n_x = Tensor.size belief.mean_x 0 in
    let n_d = Tensor.size belief.mean_d 0 in
    let valid_var_x = Tensor.size belief.var_x 0 = n_x && Tensor.size belief.var_x 1 = n_x in
    let valid_var_d = Tensor.size belief.var_d 0 = n_d && Tensor.size belief.var_d 1 = n_d in
    let valid_cov = Tensor.size belief.cov_xd 0 = n_x && Tensor.size belief.cov_xd 1 = n_d in
    valid_var_x && valid_var_d && valid_cov

  let create_belief_structure ~mean_x ~mean_d ~var_x ~var_d ~cov_xd =
    let initial = {
      mean_x;
      mean_d;
      var_x;
      var_d;
      cov_xd;
      inner_product_space = LinearAlgebra.InnerProduct.euclidean;
    } in
    
    if not (validate_dimensions initial) then
      failwith "Invalid dimensions in belief structure"
    else
      let var_x = LinearAlgebra.nearest_positive_definite var_x in
      let var_d = LinearAlgebra.nearest_positive_definite var_d in
      {initial with var_x; var_d}

  let adjusted_expectation belief d =
    let d_centered = Tensor.sub d belief.mean_d in
    let var_d_inv = LinearAlgebra.pseudo_inverse belief.var_d in
    let adjustment = Tensor.matmul 
      (Tensor.matmul belief.cov_xd var_d_inv) 
      d_centered in
    Tensor.add belief.mean_x adjustment

  let adjusted_variance belief =
    let var_d_inv = LinearAlgebra.pseudo_inverse belief.var_d in
    let term = Tensor.matmul 
      (Tensor.matmul belief.cov_xd var_d_inv)
      (Tensor.transpose_matrix belief.cov_xd) in
    let raw_var = Tensor.sub belief.var_x term in
    LinearAlgebra.nearest_positive_definite raw_var

  module Sequential = struct
    type update_sequence = {
      updates: Tensor.t list;
      initial_belief: belief_structure;
    }

    let create_sequence ~initial_belief ~updates = {
      updates;
      initial_belief;
    }

    let apply_sequence sequence =
      List.fold_left
        (fun (curr_exp, curr_var) d ->
          let curr_belief = {sequence.initial_belief with 
            mean_x = curr_exp;
            var_x = curr_var} in
          (adjusted_expectation curr_belief d,
           adjusted_variance curr_belief))
        (sequence.initial_belief.mean_x, sequence.initial_belief.var_x)
        sequence.updates
  end

  module InnerProductSpace = struct
    type t = {
      base_space: LinearAlgebra.InnerProduct.t;
      adjusted: bool;
      adjustment_operator: Tensor.t option;
    }

    let create ?(adjusted=false) ?(adjustment_operator=None) base_space =
      {base_space; adjusted; adjustment_operator}

    let compute_inner_product space x y =
      match (space.adjusted, space.adjustment_operator) with
      | false, _ -> space.base_space.compute x y
      | true, Some adj ->
          let x_adj = Tensor.sub x (Tensor.matmul adj x) in
          let y_adj = Tensor.sub y (Tensor.matmul adj y) in
          space.base_space.compute x_adj y_adj
      | true, None ->
          failwith "Adjusted space requires adjustment operator"

    let norm space x =
      Tensor.sqrt (compute_inner_product space x x)

    let distance space x y =
      let diff = Tensor.sub x y in
      norm space diff
  end

  module Diagnostics = struct
    type diagnostic_result = {
      correlation_valid: bool;
      variance_positive: bool;
      dimension_valid: bool;
      condition_numbers: float list;
    }

    let check_belief_structure belief =
      let correlation = LinearAlgebra.Correlation.compute_correlation_matrix belief.cov_xd in
      let correlation_valid = 
        LinearAlgebra.Correlation.is_valid_correlation correlation in
      
      let var_x_pd = LinearAlgebra.is_positive_definite belief.var_x in
      let var_d_pd = LinearAlgebra.is_positive_definite belief.var_d in
      
      let var_x_decomp = LinearAlgebra.Decomposition.compute_spectral belief.var_x in
      let var_d_decomp = LinearAlgebra.Decomposition.compute_spectral belief.var_d in
      
      {
        correlation_valid;
        variance_positive = var_x_pd && var_d_pd;
        dimension_valid = validate_dimensions belief;
        condition_numbers = [var_x_decomp.condition_number; 
                           var_d_decomp.condition_number];
      }
  end
end

module Generalized = struct
  module Spaces = struct
    type t = {
      compute: Tensor.t -> Tensor.t -> Tensor.t;
      gradient: Tensor.t -> Tensor.t -> Tensor.t;
      is_symmetric: bool;
      satisfies_triangle: bool;
    }

    let kl_divergence () = {
      compute = (fun p q ->
        let log_ratio = Tensor.sub (Tensor.log p) (Tensor.log q) in
        Tensor.mean (Tensor.mul p log_ratio));
      gradient = (fun p q ->
        let log_ratio = Tensor.sub (Tensor.log p) (Tensor.log q) in
        Tensor.add log_ratio (Tensor.ones_like log_ratio));
      is_symmetric = false;
      satisfies_triangle = false;
    }

    let alpha_divergence alpha = {
      compute = (fun p q ->
        let alpha_tensor = Tensor.full_like p alpha in
        let term1 = Tensor.pow p alpha_tensor in
        let term2 = Tensor.pow q 
          (Tensor.sub (Tensor.ones_like alpha_tensor) alpha_tensor) in
        Tensor.mean (Tensor.sub term1 term2));
      gradient = (fun p q ->
        let alpha_tensor = Tensor.full_like p alpha in
        let term = Tensor.pow p 
          (Tensor.sub alpha_tensor (Tensor.ones_like alpha_tensor)) in
        Tensor.mul alpha_tensor term);
      is_symmetric = (abs_float (alpha -. 0.5) < 1e-6);
      satisfies_triangle = false;
    }

    let total_variation () = {
      compute = (fun p q ->
        let diff = Tensor.sub p q in
        Tensor.mean (Tensor.abs diff) |> Tensor.mul_scalar 0.5);
      gradient = (fun p q ->
        let diff = Tensor.sub p q in
        Tensor.sign diff |> Tensor.mul_scalar 0.5);
      is_symmetric = true;
      satisfies_triangle = true;
    }
  end

  module Spaces = struct
    type t = {
      project: Tensor.t -> Tensor.t;
      is_member: Tensor.t -> bool;
      is_convex: bool;
      dimension: int;
    }

    let probability_simplex dim = {
      project = (fun x ->
        let x = Tensor.relu x in
        let sum = Tensor.sum x ~dim:[0] ~keepdim:true in
        Tensor.div x sum);
      is_member = (fun x ->
        let non_neg = Tensor.min x |> Tensor.to_float0_exn >= 0. in
        let sums_to_one = 
          abs_float (Tensor.sum x |> Tensor.to_float0_exn -. 1.) < 1e-6 in
        non_neg && sums_to_one);
      is_convex = true;
      dimension = dim;
    }

    let bounded dim lower upper = {
      project = (fun x ->
        let x = Tensor.maximum x (Tensor.full_like x lower) in
        Tensor.minimum x (Tensor.full_like x upper));
      is_member = (fun x ->
        let above_lower = Tensor.min x |> Tensor.to_float0_exn >= lower in
        let below_upper = Tensor.max x |> Tensor.to_float0_exn <= upper in
        above_lower && below_upper);
      is_convex = true;
      dimension = dim;
    }
  end

  type inference_params = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
  }

  (* Core GBI system *)
  type t = {
    solution_space: Spaces.t;
    divergence: Spaces.t;
    polish_space: Spaces.Polish.t;
  }

  (* Create standard GBI system *)
  let create ~dim ~divergence ~solution_space = {
    solution_space;
    divergence;
    polish_space = Spaces.Polish.create_with_basis dim;
  }

  module Optimization = struct
    type optimization_state = {
      point: Tensor.t;
      value: float;
      gradient_norm: float;
      iteration: int;
    }

    let adam ~learning_rate ~beta1 ~beta2 ~epsilon =
      let m = ref (Tensor.zeros [1]) in
      let v = ref (Tensor.zeros [1]) in
      let t = ref 0 in
      fun gradient ->
        t := !t + 1;
        let t_float = float_of_int !t in
        
        (* Update biased first moment estimate *)
        m := Tensor.add 
          (Tensor.mul_scalar !m beta1)
          (Tensor.mul_scalar gradient (1. -. beta1));
        
        (* Update biased second moment estimate *)
        v := Tensor.add
          (Tensor.mul_scalar !v beta2)
          (Tensor.mul_scalar (Tensor.mul gradient gradient) (1. -. beta2));
        
        (* Compute bias-corrected estimates *)
        let m_hat = Tensor.div !m (Tensor.scalar_tensor (1. -. (beta1 ** t_float))) in
        let v_hat = Tensor.div !v (Tensor.scalar_tensor (1. -. (beta2 ** t_float))) in
        
        (* Compute update *)
        Tensor.div
          (Tensor.mul_scalar m_hat learning_rate)
          (Tensor.add (Tensor.sqrt v_hat) (Tensor.scalar_tensor epsilon))

    let gradient_descent_momentum ~learning_rate ~momentum =
      let velocity = ref (Tensor.zeros [1]) in
      fun gradient ->
        velocity := Tensor.add
          (Tensor.mul_scalar !velocity momentum)
          (Tensor.mul_scalar gradient learning_rate);
        !velocity
  end

  let optimize ~params ~system ~prior ~data ~init =
    let optimizer = Optimization.adam
      ~learning_rate:params.learning_rate
      ~beta1:0.9
      ~beta2:0.999
      ~epsilon:1e-8 in

    let rec iterate current_point iter =
      if iter >= params.max_iter then current_point
      else
        (* Compute gradient *)
        let loss_grad = system.divergence.gradient current_point data in
        let prior_grad = system.divergence.gradient current_point prior in
        let total_grad = Tensor.add loss_grad prior_grad in
        
        (* Update point *)
        let update = optimizer total_grad in
        let new_point = Tensor.sub current_point update in
        
        (* Project onto solution space *)
        let projected = system.solution_space.project new_point in
        
        (* Check convergence *)
        let diff = Tensor.norm (Tensor.sub projected current_point) in
        if Tensor.to_float0_exn diff < params.tolerance then projected
        else iterate projected (iter + 1) in
    
    iterate init 0

  let infer ~system ~prior ~data ~params =
    let init = system.solution_space.project prior in
    optimize ~params ~system ~prior ~data ~init

  let infer_with_loss ~system ~prior ~data ~loss_fn ~params =
    let custom_divergence = {
      Spaces.compute = loss_fn;
      gradient = (fun p q ->
        let epsilon = 1e-6 in
        let loss = loss_fn p q in
        let grad = Tensor.grad_of_fn 
          (fun x -> loss_fn x q)
          ~wrt:[p]
          loss in
        List.hd grad);
      is_symmetric = false;
      satisfies_triangle = false;
    } in
    let system = { system with divergence = custom_divergence } in
    infer ~system ~prior ~data ~params
end

module ConjugateFamilies = struct
  type distribution_family = 
    | Gaussian
    | Gamma
    | Beta
    | Poisson
    | Binomial

  type sufficient_statistics = {
    compute: Tensor.t -> Tensor.t;
    dim: int;
  }

  type natural_parameters = {
    to_natural: Tensor.t -> Tensor.t;
    from_natural: Tensor.t -> Tensor.t;
    valid_space: Tensor.t -> bool;
  }

  let gaussian_parameters () = {
    to_natural = (fun params ->
      let mu = Tensor.select params 0 0 in
      let sigma = Tensor.select params 0 1 in
      let eta1 = Tensor.div mu (Tensor.mul sigma sigma) in
      let eta2 = Tensor.div (Tensor.neg (Tensor.ones_like sigma)) 
        (Tensor.mul (Tensor.scalar_tensor 2.) sigma) in
      Tensor.stack [eta1; eta2] ~dim:0);
    from_natural = (fun eta ->
      let eta1 = Tensor.select eta 0 0 in
      let eta2 = Tensor.select eta 0 1 in
      let sigma = Tensor.div (Tensor.neg (Tensor.ones_like eta2)) 
        (Tensor.mul (Tensor.scalar_tensor 2.) eta2) in
      let mu = Tensor.mul (Tensor.div eta1 eta2) sigma in
      Tensor.stack [mu; sigma] ~dim:0);
    valid_space = (fun eta ->
      let eta2 = Tensor.select eta 0 1 in
      Tensor.lt_scalar eta2 0.0 |> Tensor.all |> Tensor.to_bool0_exn);
  }

  let gaussian_statistics = {
    compute = (fun x ->
      let x2 = Tensor.mul x x in
      Tensor.stack [x; x2] ~dim:0);
    dim = 2;
  }

  let verify_posterior_linearity family data prior =
    match family with
    | Gaussian ->
        let params = gaussian_parameters () in
        let stats = gaussian_statistics in
        let sufficient_stats = stats.compute data in
        let natural_prior = params.to_natural prior in
        
        let posterior_params data_scale =
          let scaled_stats = Tensor.mul_scalar sufficient_stats data_scale in
          let natural_posterior = Tensor.add natural_prior scaled_stats in
          params.from_natural natural_posterior in
        
        let post1 = posterior_params 1.0 in
        let post2 = posterior_params 2.0 in
        let post_half = posterior_params 0.5 in
        
        let diff1 = Tensor.sub (Tensor.mul_scalar post1 2.0) post2 in
        let diff2 = Tensor.sub 
          (Tensor.add post1 post_half) 
          (Tensor.mul_scalar post1 1.5) in
        
        let threshold = 1e-6 in
        Tensor.norm diff1 |> Tensor.to_float0_exn < threshold &&
        Tensor.norm diff2 |> Tensor.to_float0_exn < threshold
    | _ -> false
end