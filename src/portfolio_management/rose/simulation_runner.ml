let run_simulation_multiple ~sim_type ~config ~n_runs =
  let results = Array.init n_runs (fun _ ->
    match sim_type with
    | `Sim1a -> Simulations.run_simulation 
                  { config with model_type = Models.PartiallyLinear }
    | `Sim1b -> Simulations.run_simulation 
                  { config with model_type = Models.GeneralizedPartiallyLinear (fun x -> sqrt x) }
    | `Sim3 -> Simulations.run_simulation 
                  { config with model_type = Models.PartiallyLinear }
  ) in
  
  let mean_bias = Array.fold_left (fun acc r -> acc +. r.bias) 0.0 results /. 
                 float_of_int n_runs in
  let mean_variance = Array.fold_left (fun acc r -> acc +. r.variance) 0.0 results /. 
                     float_of_int n_runs in
  let mean_mse = Array.fold_left (fun acc r -> acc +. r.mse) 0.0 results /. 
                 float_of_int n_runs in
  let coverage = Array.fold_left (fun acc r -> acc +. r.coverage) 0.0 results /. 
                float_of_int n_runs in
  
  { bias = mean_bias;
    variance = mean_variance;
    mse = mean_mse;
    coverage = coverage;
    theta_hat = config.theta_true +. mean_bias }