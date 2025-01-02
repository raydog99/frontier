let compare_estimators data theta =
  let default_params = {
    max_depth = 10;
    mtry = 3;
    min_node_size = 5;
    n_trees = 100;
    sample_fraction = 0.632;
  } in
  
  let tuned_params = ParameterTuning.optimize_parameters data [||] {
    max_depth_range = [5; 10; 15];
    mtry_range = [2; 3; 4];
    n_trees_range = [100; 200; 300];
    min_node_size_range = [5; 10; 20];
  } 10 in
  
  (* Unweighted estimator variance *)
  let unw_forest = RoseRandomForest.create_rose_forest 
    data [||] theta default_params in
  let unw_est = RoseForest.estimate data unw_forest 0.05 in
  
  (* ROSE estimator variance *)
  let rose_forest = RoseRandomForest.create_rose_forest 
    data [||] theta tuned_params in
  let rose_est = RoseForest.estimate data rose_forest 0.05 in
  
  (* Locally efficient estimator variance *)
  let loce_forest = RoseRandomForest.create_rose_forest 
    data [||] theta default_params in
  let loce_est = RoseForest.estimate_locally_efficient data loce_forest 0.05 in
  
  let v_loce_unw = loce_est.sigma_hat ** 2.0 /. unw_est.sigma_hat ** 2.0 in
  let v_unw_rose = unw_est.sigma_hat ** 2.0 /. rose_est.sigma_hat ** 2.0 in
  
  (v_loce_unw, v_unw_rose)