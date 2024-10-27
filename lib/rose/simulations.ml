let generate_simulation1a config =
  let n = config.n_samples in
  let d = config.n_dims in
  
  (* Generate covariance matrix *)
  let cov = Array.make_matrix d d 0.0 in
  for i = 0 to d - 1 do
    for j = 0 to d - 1 do
      cov.(i).(j) <- 0.9 ** (abs (i - j))
    done
  done;
  
  (* Generate data *)
  let data = Array.init n (fun _ ->
    (* Generate Z *)
    let z = Distributions.multivariate_normal (Array.make d 0.0) cov 1 in
    
    (* Generate B and X *)
    let p_z = min 1.0 (max 0.01 (tanh (3.0 *. z.(0).(0)))) in
    let b = if Random.float 1.0 < p_z then 1.0 else 0.0 in
    
    let m_x = Array.init 5 (fun j -> 
      1.0 /. (1.0 +. exp(-.z.(0).(j)))
    ) |> Array.fold_left (+.) 0.0 in
    
    let sigma_x = sqrt (b /. p_z) in
    let x = m_x +. sigma_x *. Random.float_gaussian () in
    
    (* Generate Y *)
    let f_z = m_x in
    let sigma_y = sqrt (b /. sqrt p_z) in
    let y = config.theta_true *. x +. f_z +. sigma_y *. Random.float_gaussian () in
    
    { x; y; z = z.(0) }
  ) in
  data

let generate_simulation1b config =
  let n = config.n_samples in
  let d = config.n_dims in
  
  let cov = Array.make_matrix d d 0.0 in
  for i = 0 to d - 1 do
    for j = 0 to d - 1 do
      cov.(i).(j) <- 0.9 ** (abs (i - j))
    done
  done;
  
  let data = Array.init n (fun _ ->
    let z = Distributions.multivariate_normal (Array.make d 0.0) cov 1 in
    
    let p_z = min 1.0 (max 0.01 (tanh (3.0 *. z.(0).(0)))) in
    let b = if Random.float 1.0 < p_z then 1.0 else 0.0 in
    
    let m_x = Array.init 5 (fun j -> 
      1.0 /. (1.0 +. exp(-.z.(0).(j)))
    ) |> Array.fold_left (+.) 0.0 in
    
    let sigma_x = sqrt (b /. p_z) in
    let x = m_x +. sigma_x *. Random.float_gaussian () in
    
    let mu = config.theta_true *. x +. m_x in
    let mu_sqrt = sqrt mu in
    let sigma2 = 0.01 *. mu_sqrt *. mu_sqrt *. b /. sqrt p_z in
    
    let y = Distributions.gamma_rv 
      (mu_sqrt *. mu_sqrt /. sigma2)
      (sigma2 /. mu_sqrt) in
    
    { x; y; z = z.(0) }
  ) in
  data

let generate_simulation3 config =
  let n = config.n_samples in
  let d = 3 in
  
  let data = Array.init n (fun _ ->
    let z = Distributions.multivariate_normal 
      (Array.make d 0.0)
      (Array.make_matrix d d (fun i j -> if i = j then 1.0 else 0.0))
      1 in
    
    (* Generate X with zero-inflation *)
    let p_nonzero = 0.9 -. 0.8 *. (1.0 /. (1.0 +. exp(-.z.(0).(0)))) in
    let x = if Random.float 1.0 > p_nonzero then 0.0
            else
              let m_x = Array.init 3 (fun j -> 
                1.0 /. (1.0 +. exp(-.z.(0).(j)))
              ) |> Array.fold_left (+.) 0.0 in
              m_x +. 0.1 *. Random.float_gaussian () in
    
    (* Generate Y *)
    let f_z = Array.init 3 (fun j -> 
      1.0 /. (1.0 +. exp(-.z.(0).(j)))
    ) |> Array.fold_left (+.) 0.0 in
    
    let sigma_y = if x = 0.0 then 0.9
                 else if x <= 1.5 then 0.4
                 else 0.1 in
    let y = config.theta_true *. x +. f_z +. sigma_y *. Random.float_gaussian () in
    
    { x; y; z = z.(0) }
  ) in
  data

let run_simulation config =
  (* Generate data *)
  let data = match config.model_type with
    | Models.PartiallyLinear -> generate_simulation1a config
    | Models.GeneralizedPartiallyLinear _ -> generate_simulation1b config in
  
  (* Cross-fitting *)
  let (folds, nuisance) = CrossFit.cross_fit data config.n_dims config.model_type in
  
  (* Create and tune forest *)
  let forest = RoseRandomForest.create_rose_forest 
    data nuisance config.theta_true
    {
      max_depth = 10;
      mtry = 3;
      min_node_size = 5;
      n_trees = 200;
      sample_fraction = 0.632;
    } in
  
  (* Get estimates *)
  let estimates = RoseForest.estimate data forest 0.05 in
  
  (* Compute results *)
  let bias = estimates.theta_hat -. config.theta_true in
  let mse = bias *. bias +. estimates.sigma_hat *. estimates.sigma_hat in
  let coverage = 
    let (ci_low, ci_high) = estimates.confidence_interval in
    ci_low <= config.theta_true && config.theta_true <= ci_high in
  
  { theta_hat = estimates.theta_hat;
    bias;
    variance = estimates.sigma_hat *. estimates.sigma_hat;
    mse;
    coverage = if coverage then 1.0 else 0.0 }