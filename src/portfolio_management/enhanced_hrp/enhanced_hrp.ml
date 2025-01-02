open Torch

type tensor = Tensor.t
type portfolio_weights = tensor
type returns = tensor
type covariance = tensor

type return_type = Simple | Logarithmic

type returns_data = {
  returns: tensor;
  valid_mask: tensor;
  dates: string array;
}

type transaction_costs = {
  fixed_cost: float;
  proportional_cost: float;
  market_impact: float;
}

type constraint_type =
  | LongOnly
  | Turnover of float
  | GroupConstraint of int array array * float array
  | Cardinality of int

type robust_method =
  | ResamplingShrinkage
  | RobustCovariance
  | WorstCase

type distance_metric =
  | Angular
  | Euclidean
  | Correlation
  | CustomMetric of (tensor -> tensor -> float)

type clustering_quality = {
  cophenetic_correlation: float;
  clustering_score: float;
  silhouette_score: float
}

type optimization_result = {
  weights: tensor;
  converged: bool;
  iterations: int;
  objective_value: float
}

type qp_result = {
  solution: tensor;
  converged: bool;
  iterations: int;
  objective_value: float
}

let calculate_returns ?(return_type=Simple) ?(handle_missing=true) prices =
  let valid_mask = gt prices (float_to_tensor 0.0) in
  
  let clean_prices = 
    if handle_missing then
      where prices valid_mask prices (ones_like prices)
    else prices
  in
  
  let prev_prices = narrow clean_prices ~dim:0 ~start:0 
                          ~length:((shape prices).(0) - 1) in
  let curr_prices = narrow clean_prices ~dim:0 ~start:1 
                          ~length:((shape prices).(0) - 1) in
  
  match return_type with
  | Simple -> 
      (div (sub curr_prices prev_prices) prev_prices,
       narrow valid_mask ~dim:0 ~start:1 ~length:((shape prices).(0) - 1))
  | Logarithmic -> 
      (log (div curr_prices prev_prices),
       narrow valid_mask ~dim:0 ~start:1 ~length:((shape prices).(0) - 1))

let expected_return weights returns =
  let mu = mean returns ~dim:[0] ~keepdim:true in
  matmul weights (transpose mu ~dim0:0 ~dim1:1)

let portfolio_risk weights covariance =
  matmul (matmul weights covariance) (transpose weights ~dim0:0 ~dim1:1)

let rebalance_portfolio ?costs current_weights target_weights prices =
  let trade_sizes = sub target_weights current_weights in
  
  match costs with
  | None -> target_weights
  | Some costs ->
      let fixed_costs = 
        mul (float_to_tensor costs.fixed_cost) 
            (gt (abs trade_sizes) (float_to_tensor 0.0)) in
      let prop_costs = 
        mul (float_to_tensor costs.proportional_cost) (abs trade_sizes) in
      let market_impact = 
        mul (float_to_tensor costs.market_impact)
            (pow (abs trade_sizes) (float_to_tensor 1.5)) in
      
      let total_costs = add (add fixed_costs prop_costs) market_impact in
      sub target_weights total_costs

let portfolio_risk_with_confidence weights covariance confidence_level =
  let base_risk = sqrt (portfolio_risk weights covariance) in
  let p = float_value (shape weights).(0) |> float in
  let chi_square_factor = 
    (* Chi-square quantile approximation *)
    let z = 1.96 *. sqrt (2.0 /. p) in  (* 95% confidence *)
    1.0 +. z
  in
  mul base_risk (float_to_tensor chi_square_factor)

let solve covariance =
  let p = (shape covariance).(0) in
  let ones = ones [p; 1] in
  
  (* Calculate Σ^(-1)1 *)
  let inv_cov = inverse covariance in
  let inv_cov_ones = matmul inv_cov ones in
  
  (* Calculate (1^T Σ^(-1)1) *)
  let denominator = matmul (transpose ones ~dim0:0 ~dim1:1) inv_cov_ones in
  
  (* Final weights = Σ^(-1)1 / (1^T Σ^(-1)1) *)
  div inv_cov_ones denominator

let solve_long_only covariance =
  let w = solve covariance in
  (* Project negative weights to zero and renormalize *)
  let w_pos = max w (zeros_like w) in
  div w_pos (sum w_pos)

let solve_robust covariance constraints robust_method =
  match robust_method with
  | ResamplingShrinkage ->
      (* Bootstrap resampling with shrinkage *)
      let n_samples = 100 in
      let solutions = Array.init n_samples (fun _ ->
        let perturbed_cov = 
          add covariance (mul (randn (shape covariance)) 
                            (float_to_tensor 0.1)) in
        solve perturbed_cov
      ) in
      let avg_solution = 
        stack (Array.to_list solutions) ~dim:0 |> 
        mean ~dim:[0] ~keepdim:true in
      avg_solution
      
  | RobustCovariance ->
      (* Use robust covariance estimator *)
      let robust_cov = 
        two_step_stein covariance in
      solve robust_cov
      
  | WorstCase ->
      (* Worst-case optimization *)
      let uncertainty_set = 
        mul (abs covariance) (float_to_tensor 0.1) in
      let worst_case_cov = 
        add covariance uncertainty_set in
      solve worst_case_cov

let solve_with_cardinality covariance max_assets =
  let p = (shape covariance).(0) in
  
  (* Binary variables for asset selection *)
  let z = zeros [p] ~kind:Uint8 in
  
  (* Branch and bound implementation *)
  let rec branch_and_bound best_sol best_val remaining_vars =
    if remaining_vars = [] then best_sol
    else
      let var = List.hd remaining_vars in
      let remaining = List.tl remaining_vars in
      
      set z [|var|] (Scalar.uint8 0);
      let sol0 = solve covariance in
      let val0 = portfolio_risk sol0 covariance in
      
      set z [|var|] (Scalar.uint8 1);
      let sol1 = solve covariance in
      let val1 = portfolio_risk sol1 covariance in
      
      (* Update best solution *)
      let (new_best_sol, new_best_val) =
        if float_value val0 < float_value val1 && 
           float_value val0 < best_val then
          (sol0, float_value val0)
        else if float_value val1 < best_val then
          (sol1, float_value val1)
        else
          (best_sol, best_val)
      in
      
      branch_and_bound new_best_sol new_best_val remaining
  in
  
  let initial_sol = solve covariance in
  let initial_val = 
    float_value (portfolio_risk initial_sol covariance) in
  branch_and_bound initial_sol initial_val (List.init p (fun i -> i))

let solve_with_turnover covariance current_weights max_turnover =
  let p = (shape covariance).(0) in
  let constraints = cat [
    (* Sum to 1 constraint *)
    ones [1; p];
    (* Turnover upper bound *)
    eye p;
    (* Turnover lower bound *)
    neg (eye p)
  ] ~dim:0 in
  
  let bounds = cat [
    ones [1; 1];  (* Sum to 1 *)
    mul (ones [p; 1]) (float_to_tensor max_turnover);  (* Upper *)
    mul (ones [p; 1]) (float_to_tensor max_turnover)   (* Lower *)
  ] ~dim:0 in
  
  solve_qp covariance (zeros [p; 1]) constraints bounds

let covariance_to_correlation covariance =
  let std_dev = sqrt (diagonal covariance) in
  let d = diag std_dev in
  let d_inv = inverse d in
  matmul (matmul d_inv covariance) d_inv

let correlation_to_distance correlation =
  sqrt (sub (ones_like correlation) correlation)

let cluster distances =
  let p = (shape distances).(0) in
  let clusters = ref (Array.init p (fun i -> [i])) in
  let active = ref (Array.make p true) in
  
  let rec cluster_step () =
    if Array.exists (fun x -> x) !active then begin
      (* Find minimum distance between active clusters *)
      let min_dist = ref max_float in
      let min_i = ref 0 in
      let min_j = ref 0 in
      
      Array.iteri (fun i is_active_i ->
        if is_active_i then
          Array.iteri (fun j is_active_j ->
            if is_active_j && i < j then
              let d = float_value (get distances [|i; j|]) in
              if d < !min_dist then begin
                min_dist := d;
                min_i := i;
                min_j := j
              end
          ) !active
      ) !active;
      
      (* Merge clusters *)
      clusters := Array.mapi (fun i cluster ->
        if i = !min_i then 
          List.append (!clusters).(!min_i) (!clusters).(!min_j)
        else if i = !min_j then []
        else cluster
      ) !clusters;
      
      active.(!min_j) <- false;
      cluster_step ()
    end
  in
  cluster_step ();
  !clusters

let quasi_diagonalize covariance clusters =
  let ordering = Array.concat clusters in
  let perm = of_int1 ordering in
  let cov_perm = index_select covariance ~dim:0 perm in
  index_select cov_perm ~dim:1 perm

let allocate covariance =
  let corr = covariance_to_correlation covariance in
  let dist = correlation_to_distance corr in
  let clusters = cluster dist in
  let quasi_diag = quasi_diagonalize covariance clusters in
  
  (* Inverse-variance portfolio within clusters *)
  let var = diagonal quasi_diag in
  let w = div (reciprocal var) (sum (reciprocal var)) in
  w

let naive_estimator returns =
  let n = (shape returns).(0) in
  let mu = mean returns ~dim:[0] ~keepdim:true in
  let centered = sub returns mu in
  div (matmul (transpose centered ~dim0:0 ~dim1:1) centered) 
      (float_to_tensor (float (n - 1)))

let linear_shrinkage sample_cov =
  let p = (shape sample_cov).(0) in
  let zeta = div (trace sample_cov) (float_to_tensor (float p)) in
  let target = mul (eye p) zeta in
  
  (* Compute optimal shrinkage intensity *)
  let diff = sub sample_cov target in
  let norm_diff = norm diff ~p:(Scalar 2) in
  let alpha = div norm_diff (norm sample_cov ~p:(Scalar 2)) in
  
  (* Compute shrinkage estimate *)
  add (mul target alpha) 
      (mul sample_cov (sub (float_to_tensor 1.0) alpha))

let nonlinear_lp_shrinkage sample_cov =
  let eigen = linalg_eigh sample_cov in
  let values = fst eigen in
  let vectors = snd eigen in
  
  let n, p = shape sample_cov |> fun s -> s.(0), s.(1) in
  let q = float p /. float n in
  
  (* Compute optimal nonlinear shrinkage intensities *)
  let xi_lp = map values ~f:(fun lambda ->
    let lambda_f = float_value lambda in
    lambda_f /. (1.0 +. q *. sqrt lambda_f)
  ) in
  
  (* Reconstruct covariance matrix *)
  let d = diag (of_float1 (Array.of_list xi_lp)) in
  matmul (matmul vectors d) (transpose vectors ~dim0:0 ~dim1:1)

let nonlinear_stein_shrinkage sample_cov =
  let eigen = linalg_eigh sample_cov in
  let values = fst eigen in
  let vectors = snd eigen in
  
  let n, p = shape sample_cov |> fun s -> s.(0), s.(1) in
  let q = float p /. float n in
  
  (* Compute Stein shrinkage intensities *)
  let xi_stein = map values ~f:(fun lambda ->
    let lambda_f = float_value lambda in
    lambda_f /. (1.0 +. 2.0 *. q *. lambda_f)
  ) in
  
  (* Reconstruct covariance matrix *)
  let d = diag (of_float1 (Array.of_list xi_stein)) in
  matmul (matmul vectors d) (transpose vectors ~dim0:0 ~dim1:1)

let alca_hierarchical sample_cov =
  (* Convert to correlation matrix *)
  let corr = covariance_to_correlation sample_cov in
  
  (* Compute distance matrix *)
  let dist = sub (ones_like corr) corr in
  
  (* Perform hierarchical clustering *)
  let clusters = cluster dist in
  
  (* Filter covariance based on clustering *)
  quasi_diagonalize sample_cov clusters

let stieltjes_transform values z q =
  let n = shape values |> Array.get_unsafe 0 in
  let sum = sum (div values (sub values (full_like values z))) in
  div sum (float_to_tensor (float n))

let ycm_estimator returns rho =
  let n, p = shape returns |> fun s -> s.(0), s.(1) in
  let mu = mean returns ~dim:[0] ~keepdim:true in
  let centered = sub returns mu in
  
  (* Initial estimate *)
  let ycm = ref (div (matmul (transpose centered ~dim0:0 ~dim1:1) centered) 
                    (float_to_tensor (float (n - 1)))) in
  
  (* Fixed point iteration *)
  let max_iter = 100 in
  let tol = 1e-6 in
  let converged = ref false in
  let iter = ref 0 in
  
  while not !converged && !iter < max_iter do
    let prev = !ycm in
    let xi = div (diagonal prev) 
                (float_to_tensor p |> 
                 mul (matmul centered (transpose centered ~dim0:0 ~dim1:1))) in
    let new_ycm = add 
      (mul (sub (float_to_tensor 1.0) (float_to_tensor rho)) 
           (div (matmul (transpose centered ~dim0:0 ~dim1:1) centered) 
                (float_to_tensor n)))
      (mul (float_to_tensor rho) (eye p)) in
    ycm := new_ycm;
    
    let diff = norm (sub prev new_ycm) ~p:(Scalar 2) in
    converged := float_value diff < tol;
    incr iter
  done;
  !ycm

let two_step_lp sample_cov =
  let alca = alca_hierarchical sample_cov in
  nonlinear_lp_shrinkage alca

let two_step_stein sample_cov =
  let alca = alca_hierarchical sample_cov in
  nonlinear_stein_shrinkage alca

let two_step_ycm sample_cov =
  let alca = alca_hierarchical sample_cov in
  let n, p = shape sample_cov |> fun s -> s.(0), s.(1) in
  let rho = float n /. float p in
  ycm_estimator sample_cov rho

let nested_hierarchical p gamma =
  let l = zeros [p; p] in
  
  (* Fill L matrix *)
  for i = 0 to p-1 do
    for j = 0 to i do
      set l [|i; j|] (float_to_tensor gamma);
    done
  done;
  
  (* Compute Σ = LL^T *)
  matmul l (transpose l ~dim0:0 ~dim1:1)

let one_factor p sigma sigma_r =
  let b = add (rand [p; 1]) (float_to_tensor 0.5) in (* U(0.5, 1.5) *)
  let systematic = float_to_tensor (sigma *. sigma) in
  let idiosyncratic = float_to_tensor (sigma_r *. sigma_r) in
  
  add (mul (matmul b (transpose b ~dim0:0 ~dim1:1)) systematic)
      (mul (eye p) idiosyncratic)

let diagonal_groups p =
  let n1 = p / 5 in     (* 20% *)
  let n2 = 2 * p / 5 in (* 40% *)
  let n3 = 2 * p / 5 in (* 40% *)
  
  let d = zeros [p] in
  (* Set eigenvalues for each group *)
  set_slice1 d ~start:0 ~end_:n1 (full [n1] 1.0);
  set_slice1 d ~start:n1 ~end_:(n1 + n2) (full [n2] 3.0);
  set_slice1 d ~start:(n1 + n2) ~end_:p (full [n3] 10.0);
  
  diag d

let solve_qp ?(max_iter=1000) ?(tol=1e-6) q c a b =
  let n = (shape q).(0) in
  
  (* Initialize primal and dual variables *)
  let x = ones [n; 1] in
  let y = ones [n; 1] in
  let z = ones [(shape a).(0); 1] in
  
  (* Barrier parameter *)
  let mu = ref 10.0 in
  let t = ref 1.0 in
  
  let rec iterate x y z iter =
    if iter >= max_iter then
      { solution = x;
        converged = false;
        iterations = iter;
        objective_value = float_value (add (matmul (transpose x ~dim0:0 ~dim1:1) 
                                                  (matmul q x)) 
                                         (matmul (transpose c ~dim0:0 ~dim1:1) x)) }
    else
      (* Compute residuals *)
      let rx = add (add (matmul q x) c) 
                  (neg (matmul (transpose a ~dim0:0 ~dim1:1) z)) in
      let ry = mul y x in
      let rz = sub (matmul a x) b in
      
      (* Check convergence *)
      let res_norm = norm (cat [rx; ry; rz] ~dim:0) ~p:(Scalar 2) in
      if float_value res_norm < tol then
        { solution = x;
          converged = true;
          iterations = iter;
          objective_value = float_value (add (matmul (transpose x ~dim0:0 ~dim1:1) 
                                                    (matmul q x)) 
                                           (matmul (transpose c ~dim0:0 ~dim1:1) x)) }
      else
        (* Compute search direction *)
        let d = diag (div (float_to_tensor 1.0) y) in
        let h = add q (mul d (float_to_tensor !t)) in
        let h_inv = inverse h in
        
        let dx = neg (matmul h_inv rx) in
        let dy = neg (div ry x) in
        let dz = neg (matmul (matmul (inverse (matmul (matmul a h_inv) 
                                                     (transpose a ~dim0:0 ~dim1:1))) rz)
                            (float_to_tensor 1.0)) in
        
        (* Line search *)
        let alpha = ref 1.0 in
        while !alpha > 1e-8 do
          let x_new = add x (mul dx (float_to_tensor !alpha)) in
          let y_new = add y (mul dy (float_to_tensor !alpha)) in
          let z_new = add z (mul dz (float_to_tensor !alpha)) in
          
          if all (gt x_new (zeros_like x_new)) &&
             all (gt y_new (zeros_like y_new)) then begin
            t := !t *. 1.5;
            iterate x_new y_new z_new (iter + 1)
          end else
            alpha := !alpha *. 0.5
        done;
        
        { solution = x;
          converged = false;
          iterations = iter;
          objective_value = float_value (add (matmul (transpose x ~dim0:0 ~dim1:1) 
                                                    (matmul q x)) 
                                           (matmul (transpose c ~dim0:0 ~dim1:1) x)) }
  in
  
  iterate x y z 0

let factor_risk_decomposition weights factor_loadings factor_covariance specific_risk =
  let systematic_risk = 
    matmul (matmul weights 
                   (matmul factor_loadings factor_covariance))
           (transpose factor_loadings ~dim0:0 ~dim1:1) in
  let specific_risk = 
    matmul (matmul weights specific_risk) 
           (transpose weights ~dim0:0 ~dim1:1) in
  (systematic_risk, specific_risk)

let marginal_risk_contributions weights covariance =
  let total_risk = sqrt (portfolio_risk weights covariance) in
  div (matmul covariance weights) total_risk

let component_risk_contributions weights covariance =
  let marginal = marginal_risk_contributions weights covariance in
  mul weights marginal

let hhi weights =
  sum (pow weights (float_to_tensor 2.0))

let leverage weights =
  sum (abs weights)

let in_sample_risk weights sample_cov =
  portfolio_risk weights sample_cov

let out_sample_risk weights in_sample_cov out_sample_cov =
  let numerator = 
    matmul (matmul weights in_sample_cov) 
           (matmul out_sample_cov (transpose weights ~dim0:0 ~dim1:1)) in
  let denominator = 
    pow (matmul weights (transpose weights ~dim0:0 ~dim1:1)) 
        (float_to_tensor 2.0) in
  div numerator denominator

let generate_random_returns ~n ~p =
  randn [n; p]

let split_data data ratio =
  let n = (shape data).(0) in
  let split_idx = int (float n *. ratio) in
  let in_sample = 
    narrow data ~dim:0 ~start:0 ~length:split_idx in
  let out_sample = 
    narrow data ~dim:0 ~start:split_idx ~length:(n - split_idx) in
  (in_sample, out_sample)

let is_positive_definite matrix =
  try
    let _ = linalg_cholesky matrix in
    true
  with _ -> false

let nearest_positive_definite matrix =
  let eigen = linalg_eigh matrix in
  let values = fst eigen in
  let vectors = snd eigen in
  let values_pos = max values (zeros_like values) in
  matmul (matmul vectors (diag values_pos)) 
         (transpose vectors ~dim0:0 ~dim1:1)