let compute_sandwich_variance data weights psi d_theta =
  let n = Array.length data in
  let log_weights = Array.map log weights in
  let max_log_weight = Array.fold_left max neg_infinity log_weights in
  
  let weighted_sum = ref 0.0 in
  let weighted_sq_sum = ref 0.0 in
  
  for i = 0 to n - 1 do
    let scaled_weight = exp(log_weights.(i) -. max_log_weight) in
    weighted_sum := !weighted_sum +. scaled_weight *. d_theta.(i);
    weighted_sq_sum := !weighted_sq_sum +. 
      scaled_weight *. scaled_weight *. psi.(i) *. psi.(i)
  done;
  
  (!weighted_sq_sum /. !weighted_sum) *. exp(-.max_log_weight)

let compute_confidence_interval theta_hat variance alpha =
  let z_score = 1.96 in (* 95% CI *)
  let margin = z_score *. sqrt variance in
  (theta_hat -. margin, theta_hat +. margin)

let solve_estimating_equations data weights nuisance_estimates init_theta max_iter tol =
  let n = Array.length data in
  let theta = ref init_theta in
  let converged = ref false in
  let iter = ref 0 in
  
  while not !converged && !iter < max_iter do
    let score = ref 0.0 in
    let info = ref 0.0 in
    
    for i = 0 to n - 1 do
      let (psi, d_theta) = Models.compute_influence 
        Models.PartiallyLinear data.(i) nuisance_estimates.(i) !theta in
      score := !score +. weights.(i) *. psi;
      info := !info +. weights.(i) *. d_theta
    done;
    
    let delta = !score /. !info in
    theta := !theta -. delta;
    converged := abs_float delta < tol;
    iter := !iter + 1
  done;
  !theta