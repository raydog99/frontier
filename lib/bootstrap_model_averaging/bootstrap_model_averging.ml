open Torch

type weight_vector = Tensor.t

type model_params = {
  n : int;
  m : int;
  monte_carlo_b : int;
}

type nested_model = {
  x : Tensor.t;
  k : int;
}

type optimization_result = {
  weights : Tensor.t;
  converged : bool;
  iterations : int;
  final_error : float;
}

type btma_result = {
  weights : Tensor.t;
  coefficients : Tensor.t;
  optimization_info : optimization_result;
  replications : int;
  valid_replications : int;
}

let validate params =
  params.Types.n > 0 && 
  params.m > 0 && 
  params.m <= params.n &&
  params.monte_carlo_b > 0

let default_params n = {
  Types.n = n;
  m = n / 2;
  monte_carlo_b = 500;
}

let safe_inverse x =
  try Some (Tensor.inverse x)
  with _ -> None

let pinverse x =
  let u, s, v = Tensor.svd x in
  let s_inv = Tensor.where 
    (Tensor.gt s (Tensor.of_float1 1e-10))
    (Tensor.div (Tensor.ones_like s) s)
    (Tensor.zeros_like s) in
  Tensor.matmul v
    (Tensor.matmul (Tensor.diag s_inv)
       (Tensor.transpose u ~dim0:0 ~dim1:1))

let generate_sequence ~x =
  let n = Tensor.size x [0] in
  let k = Tensor.size x [1] in
  List.init k (fun i ->
    let length = i + 1 in
    {
      x = Tensor.narrow x ~dim:1 ~start:0 ~length;
      k = length
    })

let get_model models index =
  List.nth models index

let count models = 
  List.length models

let generate_resampling_matrix ~n ~m =
  let p = Tensor.zeros [m; n] in
  for i = 0 to m - 1 do
    let idx = Random.int n in
    Tensor.set p [i; idx] (Tensor.of_float1 1.0)
  done;
  p

let calculate_ls_estimator ~x_star ~y_star =
  let xt_x = Tensor.matmul 
    (Tensor.transpose x_star ~dim0:0 ~dim1:1) x_star in
  let xt_x_inv = Tensor.inverse xt_x in
  Tensor.matmul 
    (Tensor.matmul xt_x_inv
       (Tensor.transpose x_star ~dim0:0 ~dim1:1))
    y_star

let generate_valid_sample ~x ~y ~m =
  let n = Tensor.size x [0] in
  let rec try_generate () =
    let indices = Array.init m (fun _ -> Random.int n) in
    let x_star = Tensor.stack 
      (Array.map (fun i -> Tensor.select x [0; i]) indices) ~dim:0 in
    
    match safe_inverse 
            (Tensor.matmul 
               (Tensor.transpose x_star ~dim0:0 ~dim1:1) x_star) with
    | None -> try_generate ()
    | Some _ -> 
        let y_star = Tensor.stack 
          (Array.map (fun i -> Tensor.select y [0; i]) indices) ~dim:0 in
        (x_star, y_star)
  in
  try_generate ()

let run_replications ~x ~y ~params ~models =
  let b = params.Types.monte_carlo_b in
  let residual_sum = ref None in
  let valid_count = ref 0 in
  
  while !valid_count < b do
    let x_star, y_star = generate_valid_sample ~x ~y ~m:params.m in
    
    let theta_stars = List.map (fun model ->
      calculate_ls_estimator 
        ~x_star:(Tensor.narrow x_star ~dim:1 ~start:0 ~length:model.Types.k)
        ~y_star
    ) models in
    
    let residuals = List.map2 (fun model theta ->
      Tensor.sub y (Tensor.matmul model.Types.x theta)
    ) models theta_stars in
    
    let residual_matrix = Tensor.stack residuals ~dim:1 in
    
    match !residual_sum with
    | None -> residual_sum := Some residual_matrix
    | Some sum -> residual_sum := Some (Tensor.add sum residual_matrix);
    incr valid_count
  done;
  
  Tensor.div (Option.get !residual_sum) 
    (Tensor.of_float1 (float_of_int b))

let project_onto_simplex v =
  let n = Tensor.size v [0] in
  let sorted = Tensor.sort v ~descending:true in
  
  let rec find_rho idx cumsum =
    if idx >= n then (n - 1, cumsum)
    else
      let vi = Tensor.float_value (Tensor.get sorted [idx]) in
      let new_sum = cumsum +. vi in
      let theta = (1.0 -. new_sum) /. float_of_int (idx + 1) in
      if theta +. vi > 0.0 then
        find_rho (idx + 1) new_sum
      else
        (idx - 1, cumsum -. vi)
  in
  
  let rho, cumsum = find_rho 0 0.0 in
  let theta = (1.0 -. cumsum) /. float_of_int (rho + 1) in
  
  Tensor.max 
    (Tensor.add v (Tensor.of_float1 theta))
    (Tensor.zeros_like v)

let optimize ~residual_matrix ~params =
  let n = params.Types.n in
  let num_models = Tensor.size residual_matrix [1] in
  let weights = Tensor.div 
    (Tensor.ones [num_models])
    (Tensor.of_float1 (float_of_int num_models)) in
  
  let max_iter = 1000 in
  let tol = 1e-8 in
  let learning_rate = 0.01 in
  
  let rec optimize_iter iter prev_weights best_error =
    if iter >= max_iter then
      { Types.weights = prev_weights;
        converged = false;
        iterations = iter;
        final_error = best_error }
    else
      let error_matrix = Tensor.matmul 
        (Tensor.transpose residual_matrix ~dim0:0 ~dim1:1)
        residual_matrix in
      let gradient = Tensor.div 
        (Tensor.matmul error_matrix prev_weights)
        (Tensor.of_float1 (float_of_int n)) in
      
      let new_weights = Tensor.sub prev_weights 
        (Tensor.mul gradient (Tensor.of_float1 learning_rate)) in
      let projected_weights = project_onto_simplex new_weights in
      
      let diff = Tensor.sub projected_weights prev_weights in
      let error = Tensor.float_value 
        (Tensor.sqrt (Tensor.sum (Tensor.pow diff (Tensor.of_float1 2.0)))) in
      
      if error < tol then
        { Types.weights = projected_weights;
          converged = true;
          iterations = iter;
          final_error = error }
      else
        optimize_iter (iter + 1) projected_weights error
  in
  
  optimize_iter 0 weights Float.max_float

let calculate_risk_decomposition ~x ~weights ~sigma_sq =
  let a_omega = Tensor.sub (Tensor.eye (Tensor.size x [0]))
    (Tensor.matmul x 
       (Tensor.matmul
          (Tensor.inverse (Tensor.matmul 
             (Tensor.transpose x ~dim0:0 ~dim1:1) x))
          (Tensor.matmul (Tensor.transpose x ~dim0:0 ~dim1:1) weights))) in
  
  let h_omega = Tensor.sub (Tensor.eye (Tensor.size x [0])) a_omega in
  let bias = Tensor.sum (Tensor.pow a_omega (Tensor.of_float1 2.0)) in
  let variance = Tensor.mul 
    (Tensor.of_float1 (2.0 *. sigma_sq))
    (Tensor.sum (Tensor.diagonal (Tensor.matmul h_omega h_omega))) in
  
  (bias, variance)

let calculate_xi_n ~x ~y =
  let n = Tensor.size x [0] in
  let xt_x = Tensor.matmul 
    (Tensor.transpose x ~dim0:0 ~dim1:1) x in
  Tensor.div xt_x (Tensor.of_float1 (float_of_int n))

let check_conditions ~x ~y ~params =
  let n = Tensor.size x [0] in
  let k = Tensor.size x [1] in
  
  (* Design matrix boundedness *)
  let c1 = Tensor.lt (Tensor.max x) 
    (Tensor.of_float1 Float.max_float) in
  
  (* Eigenvalue conditions *)
  let xt_x = Tensor.matmul 
    (Tensor.transpose x ~dim0:0 ~dim1:1) x in
  let min_eigen = Tensor.min 
    (Tensor.symeig xt_x ~eigenvectors:false) in
  let c2 = Tensor.gt min_eigen (Tensor.of_float1 0.0) in
  
  (* Minimum eigenvalue condition *)
  let x_star, _ = generate_valid_sample 
    ~x ~y ~m:params.m in
  let xt_star_x_star = Tensor.matmul 
    (Tensor.transpose x_star ~dim0:0 ~dim1:1) x_star in
  let c3 = Tensor.gt 
    (Tensor.min (Tensor.symeig xt_star_x_star ~eigenvectors:false))
    (Tensor.of_float1 0.0) in
  
  (* Growth rate condition *)
  let c4 = float_of_int k < sqrt (float_of_int n) in
  
  (Tensor.float_value c1, 
   Tensor.float_value c2, 
   Tensor.float_value c3,
   c4)

let calculate_divergence ~mu_hat ~mu =
  let diff = Tensor.sub mu_hat mu in
  Tensor.sum (Tensor.pow diff (Tensor.of_float1 2.0))

let asymptotic_optimality_condition ~x ~y ~weights ~sigma_sq ~params =
  let divergence = calculate_divergence 
    ~mu_hat:(Tensor.matmul x weights) ~mu:y in
  
  let inf_divergence = ref Float.max_float in
  let num_trials = 1000 in
  
  for _ = 1 to num_trials do
    let random_weights = project_onto_simplex 
      (Tensor.randn [Tensor.size weights [0]]) in
    let curr_divergence = calculate_divergence
      ~mu_hat:(Tensor.matmul x random_weights) ~mu:y in
    inf_divergence := min !inf_divergence (Tensor.float_value curr_divergence)
  done;
  
  let ratio = Tensor.float_value divergence /. !inf_divergence in
  abs(ratio -. 1.0) < 0.01

let calculate_h_omega ~x ~weights =
  Tensor.matmul x 
    (Tensor.matmul
       (Tensor.inverse (Tensor.matmul 
          (Tensor.transpose x ~dim0:0 ~dim1:1) x))
       (Tensor.matmul (Tensor.transpose x ~dim0:0 ~dim1:1) weights))

let estimate ~x ~y ~params =
  let models = generate_sequence ~x in
  let residual_matrix = run_replications 
    ~x ~y ~params ~models in
  
  let opt_result = optimize 
    ~residual_matrix ~params in
  
  let coefficients = List.fold_left2
    (fun acc model weight ->
      let theta = calculate_ls_estimator 
        ~x_star:model.Types.x
        ~y_star:y in
      Tensor.add acc 
        (Tensor.mul theta (Tensor.select opt_result.weights [weight]))
    )
    (Tensor.zeros [Tensor.size x [1]; 1])
    models
    (List.init (List.length models) (fun i -> i)) in
  
  { Types.weights = opt_result.weights;
    coefficients;
    optimization_info = opt_result;
    replications = params.monte_carlo_b;
    valid_replications = params.monte_carlo_b }

let multinomial_properties_condition ~pi_matrix ~n ~m =
  (* Expected properties *)
  let expected_mean = float_of_int m /. float_of_int n in
  let expected_var = float_of_int m *. float_of_int (n - 1) /. 
                    (float_of_int n ** 2.0) in
  
  (* Calculate empirical properties *)
  let empirical_mean = Tensor.mean pi_matrix in
  let empirical_var = Tensor.var pi_matrix ~unbiased:true in
  
  (* Verify properties within tolerance *)
  let mean_ok = abs_float (Tensor.float_value empirical_mean -. expected_mean) < 0.1 in
  let var_ok = abs_float (Tensor.float_value empirical_var -. expected_var) < 0.1 in
  
  (mean_ok, var_ok)