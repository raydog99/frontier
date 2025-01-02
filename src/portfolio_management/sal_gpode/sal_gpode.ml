open Torch

type gp_params = {
  lengthscales: Tensor.t;
  signal_variance: float;
  noise_variance: float;
}

type sparse_gp = {
  inducing_inputs: Tensor.t;
  inducing_outputs: Tensor.t;
  variational_mean: Tensor.t;
  variational_cov: Tensor.t;
  params: gp_params;
}

type gpode_model = {
  gp: sparse_gp;
  integration_times: Tensor.t;
  dt: float;
}

let predict_gp ~model ~x =
  let gp = model.gp in
  let k_star = kernel_matrix 
    ~params:gp.params 
    ~x1:x 
    ~x2:gp.inducing_inputs in
  let k_star_k_inv = Tensor.(
    matmul k_star (inverse gp.variational_cov)
  ) in
  let mean = Tensor.(
    matmul k_star_k_inv gp.variational_mean
  ) in
  let var = Tensor.(
    let k_xx = kernel_matrix ~params:gp.params ~x1:x ~x2:x in
    let term2 = matmul k_star_k_inv (transpose k_star ~dim0:0 ~dim1:1) in
    k_xx - term2 + float_tensor [gp.params.noise_variance]
  ) in
  mean, var

let calculate_log_marginal_likelihood ~model =
  let gp = model.gp in
  let k = kernel_matrix 
    ~params:gp.params 
    ~x1:gp.inducing_inputs 
    ~x2:gp.inducing_inputs in
  
  let noise_term = Tensor.(eye (size k 0) * float_tensor [gp.params.noise_variance]) in
  let k_noisy = Tensor.(k + noise_term) in
  
  let n = float_of_int (Tensor.size gp.inducing_outputs 0) in
  let det_term = Tensor.logdet k_noisy in
  let y = gp.inducing_outputs in
  
  let solve = Tensor.triangular_solve y (Tensor.cholesky k_noisy) ~upper:false in
  let quad_term = Tensor.(matmul (transpose solve ~dim0:0 ~dim1:1) solve) in
  
  Tensor.(
    neg (det_term + quad_term + float_tensor [n *. log (2. *. Float.pi)]) * float_tensor [0.5]
  )

let rbf_kernel ~params ~x1 ~x2 =
  let sq_dist = Tensor.( 
    let scaled = x1 / params.lengthscales - x2 / params.lengthscales in
    sum (scaled * scaled) ~dim:[-1] 
  ) in
  Tensor.(params.signal_variance * exp (neg sq_dist *. (-0.5)))

let kernel_matrix ~params ~x1 ~x2 =
  let n1 = Tensor.size x1 0 in
  let n2 = Tensor.size x2 0 in
  let x1_exp = Tensor.expand x1 ~size:[n1; n2; -1] in
  let x2_exp = Tensor.expand x2 ~size:[n1; n2; -1] in
  rbf_kernel ~params ~x1:x1_exp ~x2:x2_exp
  
let nystrom_approximation ~kernel ~inducing_points ~x =
  let k_mm = kernel inducing_points inducing_points in
  let k_nm = kernel x inducing_points in
  let l = Tensor.cholesky k_mm in
  let v = Tensor.triangular_solve k_nm l ~upper:false in
  Tensor.matmul v (Tensor.transpose v ~dim0:0 ~dim1:1)

let random_feature_approximation ~num_features ~input_dim ~params =
  let omega = Tensor.randn [num_features; input_dim] in
  let b = Tensor.(2. *. Float.pi *. randn [num_features; 1]) in
  
  fun x ->
    let scaled_x = Tensor.(x / params.lengthscales) in
    let projection = Tensor.(
      matmul omega scaled_x + b
    ) in
    Tensor.(cos projection * float_tensor [sqrt (2. /. float_of_int num_features)])
  
let sample_function ~phi ~weights x =
  let features = phi x in
  Tensor.(matmul (transpose features ~dim0:0 ~dim1:1) weights)

module Integration = struct
  type integrator = [ `RK4 | `Adaptive | `DormandPrince ]
  
  let integrate_trajectory ~model ~x0 ~t_span ~integrator ~tol =
    let rec integrate t x trajs dt =
      if t >= t_span then
        List.rev trajs
      else
        let dx = match integrator with
          | `RK4 -> 
            (* Standard RK4 step *)
            let f x_t = 
              let pred_mean, _ = predict_gp ~model ~x:x_t in
              pred_mean
            in
            let k1 = f x in
            let k2 = f Tensor.(x + k1 * (float_tensor [dt /. 2.])) in
            let k3 = f Tensor.(x + k2 * (float_tensor [dt /. 2.])) in
            let k4 = f Tensor.(x + k3 * (float_tensor [dt])) in
            Tensor.((k1 + k2 * 2. + k3 * 2. + k4) * (float_tensor [dt /. 6.]))
          | `Adaptive | `DormandPrince ->
            (* Adaptive step size control *)
            let f = predict_gp ~model in
            let pred_mean, pred_var = f x in
            let x_half = Tensor.(x + pred_mean * float_tensor [dt /. 2.]) in
            let _, var_half = f x_half in
            let new_dt = 
              if Tensor.(mean pred_var |> float_value) > tol
              then dt *. 0.5
              else dt *. 1.5
            in
            Tensor.(pred_mean * float_tensor [new_dt])
        in
        let x_new = Tensor.(x + dx) in
        integrate (t +. dt) x_new (x_new :: trajs) dt
    in
    integrate 0. x0 [x0] (t_span /. 100.)

  let sample_trajectories ~model ~x0 ~num_samples ~integrator ~tol =
    List.init num_samples (fun _ ->
      integrate_trajectory ~model ~x0 ~t_span:(Tensor.float_value (Tensor.max model.integration_times))
        ~integrator ~tol
    )
end

module Information = struct
  type t = [ `MutualInfo | `Entropy | `Variance | `ELBO ]
  
  let calculate_entropy ~trajectories =
    let k = Tensor.size trajectories 0 in 
    let mu = Tensor.mean trajectories ~dim:[0] in
    let centered = Tensor.(trajectories - expand mu ~size:[k; -1; -1]) in
    let cov = Tensor.(
      matmul 
        (transpose centered ~dim0:1 ~dim1:2)
        centered
    ) in
    let det = Tensor.(logdet cov |> float_value) in
    0.5 *. det +. float_of_int k *. 0.5 *. log (2. *. Float.pi)

  let calculate_mutual_info ~model ~x0 ~num_samples =
    let trajectories = Integration.sample_trajectories 
      ~model ~x0 ~num_samples ~integrator:`DormandPrince ~tol:1e-6 in
    let stacked = Tensor.stack trajectories ~dim:0 in
    calculate_entropy ~trajectories:stacked

  let calculate_variance ~model ~x0 =
    let _, var = predict_gp ~model ~x:x0 in
    Tensor.(sum var |> float_value)

  let calculate_elbo ~model ~x0 ~trajectories =
    let gp = model.gp in
    
    (* KL divergence between q(U) and p(U) *)
    let kl_div = 
      let prior_cov = kernel_matrix 
        ~params:gp.params 
        ~x1:gp.inducing_inputs 
        ~x2:gp.inducing_inputs in
      let mu = gp.variational_mean in
      let sigma = gp.variational_cov in
      
      Tensor.(
        let term1 = logdet prior_cov in
        let term2 = neg (logdet sigma) in
        let term3 = float_of_int (size mu 0) in
        let term4 = trace (matmul (inverse prior_cov) sigma) in
        let term5 = matmul 
          (matmul (transpose mu ~dim0:0 ~dim1:1) (inverse prior_cov))
          mu in
        
        (term1 +. term2 -. term3 +. term4 +. term5) *. 0.5
        |> float_value
      ) in

    (* Log likelihood *)
    let log_likelihood =
      let pred_mean, pred_var = predict_gp ~model ~x:trajectories in
      let diff = Tensor.(trajectories - pred_mean) in
      Tensor.(
        let term1 = size trajectories 0 |> float_of_int |> ( *. ) (log (2. *. Float.pi)) in
        let term2 = sum (log pred_var) |> float_value in
        let term3 = sum (diff * diff / pred_var) |> float_value in
        
        -.(term1 +. term2 +. term3) *. 0.5
      ) in
    
    log_likelihood -. kl_div
end

module Safety = struct
  type constraint_type = [ `Hard | `Soft of float ]
  
  type config = {
    x_min: Tensor.t;
    x_max: Tensor.t;
    constraint_type: constraint_type;
    prob_threshold: float;
  }

  let evaluate_safety ~model ~x0 ~config ~num_samples =
    let trajectories = Integration.sample_trajectories 
      ~model ~x0 ~num_samples ~integrator:`DormandPrince ~tol:1e-6 in
    
    let stacked = Tensor.stack trajectories ~dim:0 in
    let within_bounds = Tensor.(
      let above_min = stacked >= config.x_min in
      let below_max = stacked <= config.x_max in
      logical_and above_min below_max
    ) in
    
    match config.constraint_type with
    | `Hard -> 
      Tensor.(mean within_bounds |> float_value) >= config.prob_threshold
    | `Soft w ->
      let violations = Tensor.(
        sum (float_tensor [1.] - within_bounds) |> float_value
      ) in
      exp (-. w *. violations)

  let compute_gradient ~model ~x ~config =
    let x_grad = Tensor.requires_grad_ x in
    let safety = evaluate_safety 
      ~model 
      ~x0:x_grad 
      ~config 
      ~num_samples:100 in
    Tensor.backward (Tensor.float_tensor [safety]);
    Tensor.grad x_grad
end

(* ... Continued from previous implementation ... *)

module TimeVaryingSafety = struct
  type constraint_fn = float -> Tensor.t * Tensor.t
  
  let evaluate_time_varying_safety ~model ~x0 ~constraint_fn ~num_samples =
    let trajectories = Integration.sample_trajectories 
      ~model ~x0 ~num_samples ~integrator:`DormandPrince ~tol:1e-6 in
    
    let stacked = Tensor.stack trajectories ~dim:0 in
    let times = model.integration_times in
    
    Tensor.map times ~f:(fun t ->
      let min_bound, max_bound = constraint_fn (Tensor.float_value t) in
      let states = Tensor.select stacked ~dim:1 ~index:(int_of_float (Tensor.float_value t)) in
      let safe = Tensor.(
        let above_min = states >= min_bound in
        let below_max = states <= max_bound in
        logical_and above_min below_max
      ) in
      Tensor.(mean safe |> float_value)
    )
    |> Tensor.min 
    |> Tensor.float_value

  let compute_time_varying_gradient ~model ~x ~t ~constraint_fn =
    let x_grad = Tensor.requires_grad_ x in
    let min_bound, max_bound = constraint_fn t in
    
    let safety = evaluate_time_varying_safety 
      ~model 
      ~x0:x_grad
      ~constraint_fn
      ~num_samples:100 in
      
    Tensor.backward (Tensor.float_tensor [safety]);
    Tensor.grad x_grad
end

module ModelUpdate = struct
  let update_variational_distribution ~model ~new_data ~num_iters =
    let x_data, y_data = new_data in
    let gp = model.gp in
    
    let rec optimize mean cov iter =
      if iter >= num_iters then (mean, cov)
      else
        let elbo = Information.calculate_elbo 
          ~model:{model with gp = {gp with 
            variational_mean = mean;
            variational_cov = cov;
          }}
          ~x0:x_data
          ~trajectories:y_data in
          
        let mean_grad = Tensor.requires_grad_ mean in
        let cov_grad = Tensor.requires_grad_ cov in
        Tensor.backward (Tensor.float_tensor [elbo]);
        
        let new_mean = Tensor.(mean + mean_grad * float_tensor [1e-3]) in
        let new_cov = Tensor.(cov + cov_grad * float_tensor [1e-3]) in
        
        optimize new_mean new_cov (iter + 1)
    in
    
    let final_mean, final_cov = optimize 
      gp.variational_mean 
      gp.variational_cov 
      0 in
      
    {model with gp = {gp with 
      variational_mean = final_mean;
      variational_cov = final_cov;
    }}

  let optimize_hyperparams ~model ~learning_rate ~num_iters =
    let gp = model.gp in
    let params = gp.params in
    
    let lengthscales = Tensor.requires_grad_ params.lengthscales in
    let signal_var = Tensor.requires_grad_ (Tensor.float_tensor [params.signal_variance]) in
    let noise_var = Tensor.requires_grad_ (Tensor.float_tensor [params.noise_variance]) in
    
    let optimizer = Optimizer.adam [lengthscales; signal_var; noise_var] ~lr:learning_rate in
    
    let rec optimize iter =
      if iter >= num_iters then
        {
          lengthscales = Tensor.no_grad lengthscales;
          signal_variance = Tensor.(float_value signal_var);
          noise_variance = Tensor.(float_value noise_var);
        }
      else
        let () = Optimizer.zero_grad optimizer in
        
        let current_params = {
          lengthscales;
          signal_variance = Tensor.(float_value signal_var);
          noise_variance = Tensor.(float_value noise_var);
        } in
        
        let marginal_ll = calculate_log_marginal_likelihood 
          ~model:{model with gp = {gp with params = current_params}} in
        
        Tensor.backward (Tensor.neg marginal_ll);
        Optimizer.step optimizer;
        optimize (iter + 1)
    in
    
    let final_params = optimize 0 in
    {model with gp = {gp with params = final_params}}
end

module SafeOptimization = struct
  type config = {
    num_samples: int;
    max_iter: int;
    learning_rate: float;
    safety_config: Safety.config;
  }

  let optimize ~model ~x_init ~config =
    let x = ref (Tensor.requires_grad_ x_init) in
    let optimizer = Optimizer.adam [!x] ~lr:config.learning_rate in
    
    let best_x = ref None in
    let best_val = ref neg_infinity in
    
    for _ = 1 to config.max_iter do
      Optimizer.zero_grad optimizer;
      
      let safety = Safety.evaluate_safety 
        ~model 
        ~x0:!x 
        ~config:config.safety_config 
        ~num_samples:config.num_samples in
        
      if safety >= config.safety_config.prob_threshold then
        let acq_value = Information.calculate_mutual_info 
          ~model 
          ~x0:!x 
          ~num_samples:config.num_samples in
          
        if acq_value > !best_val then begin
          best_val := acq_value;
          best_x := Some (Tensor.copy !x)
        end;
        
        Tensor.backward (Tensor.float_tensor [-.acq_value]);
        Optimizer.step optimizer
    done;
    !best_x

  let optimize_with_time_varying_constraints ~model ~x_init ~constraint_fn ~config =
    let x = ref (Tensor.requires_grad_ x_init) in
    let optimizer = Optimizer.adam [!x] ~lr:config.learning_rate in
    
    let rec optimize iter best_x best_val =
      if iter >= config.max_iter then best_x
      else
        let () = Optimizer.zero_grad optimizer in
        
        let safety = TimeVaryingSafety.evaluate_time_varying_safety
          ~model 
          ~x0:!x 
          ~constraint_fn 
          ~num_samples:config.num_samples in
          
        if safety >= config.safety_config.prob_threshold then
          let acq_value = Information.calculate_mutual_info 
            ~model 
            ~x0:!x 
            ~num_samples:config.num_samples in
            
          if acq_value > best_val then begin
            Tensor.backward (Tensor.float_tensor [-.acq_value]);
            Optimizer.step optimizer;
            optimize (iter + 1) (Tensor.copy !x) acq_value
          end else
            optimize (iter + 1) best_x best_val
        else
          optimize (iter + 1) best_x best_val
    in
    
    optimize 0 x_init neg_infinity
end

let create_model ~inducing_points ~params ~dt ~t_span =
  let gp = {
    inducing_inputs = inducing_points;
    inducing_outputs = Tensor.zeros_like inducing_points;
    variational_mean = Tensor.zeros_like inducing_points;
    variational_cov = Tensor.eye (Tensor.size inducing_points 0);
    params;
  } in
  
  {
    gp;
    integration_times = Tensor.linspace ~start:0. ~end_:t_span ~steps:100;
    dt;
  }

let safe_active_learning ~model ~config =
  let rec iterate model history iter =
    if iter >= config.SafeOptimization.max_iter then
      Ok (model, history)
    else
      match SafeOptimization.optimize ~model ~x_init:(Tensor.randn [2]) ~config with
      | None -> Error "Could not find safe point"
      | Some x -> 
        let trajectories = Integration.sample_trajectories 
          ~model 
          ~x0:x 
          ~num_samples:config.num_samples 
          ~integrator:`DormandPrince 
          ~tol:1e-6 in
          
        let traj_tensor = Tensor.stack trajectories ~dim:0 in
        
        let updated_model = ModelUpdate.update_variational_distribution 
          ~model 
          ~new_data:(x, traj_tensor)
          ~num_iters:100 in
          
        let performance = Information.calculate_mutual_info 
          ~model:updated_model 
          ~x0:x 
          ~num_samples:config.num_samples in
          
        iterate updated_model (performance :: history) (iter + 1)
  in
  iterate model [] 0