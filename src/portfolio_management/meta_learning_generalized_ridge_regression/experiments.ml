open Torch
open Types
open Matrix_ops

let run_trial ~config ~omega ~sigma ~estimation_method =
  let tasks = Data_gen.generate_tasks ~config ~omega ~sigma in
  
  let test_task = Data_gen.generate_task_data 
    ~dim:config.dim
    ~num_samples:config.num_test_samples
    ~omega ~sigma
    ~sigma_sq:config.sigma_sq in
    
  let omega_hat = estimation_method tasks in
  
  let identity_risk = 
    Ridge.oracle_risk 
      ~x_new:test_task.x 
      ~y_new:test_task.y 
      ~beta:(Ridge.estimate ~x:test_task.x ~y:test_task.y 
                           ~a:(Tensor.eye config.dim) 
                           ~n_l:config.num_test_samples)
      ~omega in
      
  let estimated_risk =
    Ridge.oracle_risk 
      ~x_new:test_task.x 
      ~y_new:test_task.y 
      ~beta:(Ridge.estimate ~x:test_task.x ~y:test_task.y 
                           ~a:omega_hat 
                           ~n_l:config.num_test_samples)
      ~omega in
      
  let limiting_risk = 
    let gamma = float_of_int config.dim /. float_of_int config.num_test_samples in
    Ridge.compute_asymptotic_risk 
      ~x:test_task.x 
      ~y:test_task.y 
      ~omega 
      ~sigma_sq:config.sigma_sq 
      ~gamma in
      
  (identity_risk, estimated_risk, limiting_risk, omega_hat)

let run_unregularized_experiment ~config ~omega ~sigma =
  let results = List.init config.num_runs (fun _ ->
    let id_risk, est_risk, lim_risk, omega_hat = 
      run_trial ~config ~omega ~sigma 
        ~estimation_method:(fun tasks ->
          Hyper_covariance.estimate_unregularized 
            ~tasks 
            ~sigma_sq:config.sigma_sq) in
    
    let frob_error = frobenius_norm (Tensor.sub omega_hat omega) in
    (id_risk, est_risk, lim_risk, frob_error)
  ) in
  
  let mean_list l = List.fold_left (+.) 0. l /. float_of_int config.num_runs in
  let std_list l =
    let m = mean_list l in
    let var = List.fold_left (fun acc x -> acc +. (x -. m) *. (x -. m)) 
                            0. l /. float_of_int config.num_runs in
    sqrt var
  in
  
  let id_risks, est_risks, lim_risks, frob_errors = List.split4 results in
  (mean_list id_risks, std_list id_risks,
   mean_list est_risks, std_list est_risks,
   mean_list lim_risks)

let run_l1_regularized_experiment ~config ~omega ~sigma ~lambda =
  let results = List.init config.num_runs (fun _ ->
    let id_risk, est_risk, lim_risk, omega_hat = 
      run_trial ~config ~omega ~sigma 
        ~estimation_method:(fun tasks ->
          Hyper_covariance.estimate_l1_regularized 
            ~tasks 
            ~sigma_sq:config.sigma_sq 
            ~lambda) in
    
    let frob_error = frobenius_norm (Tensor.sub omega_hat omega) in
    (id_risk, est_risk, lim_risk, frob_error)
  ) in
  
  let id_risks, est_risks, lim_risks, frob_errors = List.split4 results in
  let mean_list = List.fold_left (+.) 0. in
  let len = float_of_int config.num_runs in
  (mean_list id_risks /. len,
   mean_list est_risks /. len,
   mean_list lim_risks /. len,
   mean_list frob_errors /. len)

let run_correlation_benchmark ~config ~lambda =
  let omega = Data_gen.generate_toeplitz ~dim:config.dim ~a:16. ~b:5. in
  let sigma = Tensor.eye config.dim in
  let tasks = Data_gen.generate_tasks ~config ~omega ~sigma in
  
  let l0 = config.num_tasks / 5 in
  
  (* Get risk for correlation-based method *)
  let omega_corr = Hyper_covariance.estimate_correlation ~tasks ~l0 ~lambda in
  let test_task = Data_gen.generate_task_data 
    ~dim:config.dim 
    ~num_samples:config.num_test_samples
    ~omega ~sigma 
    ~sigma_sq:config.sigma_sq in
  
  let corr_risk = Ridge.oracle_risk
    ~x_new:test_task.x 
    ~y_new:test_task.y
    ~beta:(Ridge.estimate 
      ~x:test_task.x ~y:test_task.y 
      ~a:omega_corr 
      ~n_l:config.num_test_samples)
    ~omega in
  
  (* Get risk for L1 regularized method *)
  let omega_l1 = Hyper_covariance.estimate_l1_regularized 
    ~tasks 
    ~sigma_sq:config.sigma_sq 
    ~lambda in
  
  let l1_risk = Ridge.oracle_risk
    ~x_new:test_task.x 
    ~y_new:test_task.y
    ~beta:(Ridge.estimate 
      ~x:test_task.x ~y:test_task.y 
      ~a:omega_l1 
      ~n_l:config.num_test_samples)
    ~omega in
  
  (corr_risk, l1_risk)