open Torch

(* Configuration parameters for the robust optimization *)
type config = {
  n : int;                      (* Number of assets *)
  t : int;                      (* Number of periods *)
  confidence : float;           (* Confidence level (1 - delta_0) *)
  learning_rate : float;        (* Learning rate for optimization *)
  max_iter : int;               (* Maximum iterations for optimization *)
  rebalance_threshold : float;  (* Threshold for portfolio rebalancing *)
  transaction_cost : float;     (* Transaction cost as a percentage *)
  taylor_order : int;           (* Order of Taylor expansion (1 or 2) *)
}

(* Default configuration *)
let default_config ~n ~t = {
  n;
  t;
  confidence = 0.95;
  learning_rate = 0.01;
  max_iter = 1000;
  rebalance_threshold = 0.05;
  transaction_cost = 0.002;
  taylor_order = 1;
}

(* Performance metrics for a portfolio strategy *)
type performance = {
  final_wealth : float;
  total_return : float;
  mean_daily_return : float;
  annualized_return : float;
  annualized_volatility : float;
  sharpe_ratio : float;
  max_drawdown : float;
  num_rebalances : int;
  total_transaction_costs : float;
}

(* Simulation results for a strategy *)
type simulation_result = {
  final_wealth : float;
  wealth_history : float array;
  rebalance_times : int list;
  total_transaction_costs : float;
  final_weights : Tensor.t;
}

(* Strategy representation *)
type strategy = {
  n : int;                              (* Number of assets *)
  periods : int;                        (* Number of periods *)
  strategy_fn : Tensor.t -> Tensor.t;   (* Function mapping returns to allocation *)
  threshold : float;                    (* Rebalancing threshold *)
  transaction_cost : float;             (* Transaction cost rate *)
}

(* Compute L2 norm of a tensor *)
let l2_norm x = 
    Tensor.norm x ~p:(Scalar.float 2.0) ~dim:[0] ~keepdim:false

(* Compute covariance matrix of a tensor *)
let covariance x =
    let mean = Tensor.mean x ~dim:[0] ~keepdim:true in
    let centered = Tensor.sub x mean in
    let n = Tensor.size centered |> List.hd |> float_of_int in
    let cov = Tensor.mm (Tensor.transpose centered ~dim0:1 ~dim1:0) centered in
    Tensor.div_scalar cov (Scalar.float (n -. 1.0))

(* Compute mean of a tensor *)
let mean x = 
    Tensor.mean x ~dim:[0] ~keepdim:false

(* Compute the squared L2 Wasserstein distance between empirical distributions *)
let w2_empirical samples1 samples2 =
    (* For empirical distributions, approximation based on means and covariances *)
    let mean1 = mean samples1 in
    let mean2 = mean samples2 in
    let cov1 = covariance samples1 in
    let cov2 = covariance samples2 in
    
    (* W2^2 ≈ ||mean1 - mean2||^2 + ||cov1 - cov2||_F *)
    let mean_diff = Tensor.sub mean1 mean2 in
    let mean_term = Tensor.dot mean_diff mean_diff in
    
    let cov_diff = Tensor.sub cov1 cov2 in
    let cov_term = Tensor.norm cov_diff ~p:(Scalar.float 2.0) ~dim:[0; 1] ~keepdim:false in
    
    Tensor.add mean_term (Tensor.mul cov_term cov_term)

(* Convert price data to returns *)
let prices_to_returns prices =
    let n = Tensor.size prices |> List.hd in
    let returns = Tensor.empty [n-1; Tensor.size prices |> List.nth 1] 
                  ~kind:(Tensor.kind prices) ~device:(Tensor.device prices) in
    
    for i = 1 to n-1 do
      let prev_prices = Tensor.slice prices ~dim:0 ~start:(i-1) ~end_:i ~step:1 in
      let curr_prices = Tensor.slice prices ~dim:0 ~start:i ~end_:(i+1) ~step:1 in
      let period_returns = Tensor.div (Tensor.sub curr_prices prev_prices) prev_prices in
      Tensor.copy_ (Tensor.slice returns ~dim:0 ~start:(i-1) ~end_:i ~step:1) ~src:period_returns
    done;
    
    returns

(* Load price data from a CSV file *)
let load_prices_from_csv filename =
    (* Read the file *)
    let lines = ref [] in
    let ic = open_in filename in
    try
      while true do
        let line = input_line ic in
        lines := line :: !lines
      done;
      assert false (* This should never be reached *)
    with End_of_file ->
      close_in ic;
      List.rev !lines
    
    (* Parse CSV data *)
    let header = List.hd !lines in
    let data = List.tl !lines in
    
    (* Extract headers (assuming first column is date) *)
    let headers = String.split_on_char ',' header |> List.tl in
    let n = List.length headers in
    
    (* Parse price data *)
    let prices = Array.make_matrix (List.length data) n 0.0 in
    
    List.iteri (fun row_idx line ->
      let values = String.split_on_char ',' line |> List.tl |> List.map float_of_string |> Array.of_list in
      for col_idx = 0 to n-1 do
        prices.(row_idx).(col_idx) <- values.(col_idx)
      done
    ) data;
    
    (* Convert to tensor *)
    let prices_tensor = Tensor.of_float2 prices in
    
    headers, prices_tensor

(* Split data into training and testing sets *)
let split_train_test prices split_ratio =
    let t = Tensor.size prices |> List.hd in
    let split_idx = int_of_float (float_of_int t *. split_ratio) in
    
    let train = Tensor.slice prices ~dim:0 ~start:0 ~end_:split_idx ~step:1 in
    let test = Tensor.slice prices ~dim:0 ~start:split_idx ~end_:t ~step:1 in
    
    train, test

(* Generate synthetic price data for testing *)
let generate_synthetic_prices ~n ~days ~volatility ~drift =
    (* Initialize price matrix *)
    let prices = Tensor.ones [days; n] ~kind:Float in
    
    (* Set initial prices between 80 and 120 *)
    for i = 0 to n-1 do
      let initial_price = 80.0 +. (Random.float 40.0) in
      Tensor.set prices [|0; i|] (Tensor.float_vec [initial_price])
    done;
    
    (* Generate price paths *)
    for day = 1 to days-1 do
      for asset = 0 to n-1 do
        let prev_price = Tensor.get prices [|day-1; asset|] |> Tensor.to_float0_exn in
        
        (* Generate random return with drift *)
        let return = drift +. volatility *. (Random.float 2.0 -. 1.0) in
        let new_price = prev_price *. (1.0 +. return) in
        
        Tensor.set prices [|day; asset|] (Tensor.float_vec [new_price])
      done
    done;
    
    prices

(* Taylor series approximation for investment strategies *)
module TaylorApproximation = struct
  (* First-order Taylor approximation of the strategy *)
  let first_order_approximation ~initial_values ~gradients ~returns =
    let n = Tensor.size returns |> List.nth 1 in  (* Number of assets *)
    let t = Tensor.size returns |> List.hd in     (* Number of time periods *)
    
    let strategy = Tensor.empty [t; n] ~kind:(Tensor.kind returns) ~device:(Tensor.device returns) in
    
    for time = 0 to t-1 do
      (* For each time point, compute the strategy based on the Taylor expansion *)
      let time_slice = Tensor.slice strategy ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
      
      (* Get the initial values for this time *)
      let init_vals = Tensor.slice initial_values ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
      
      (* Get the relevant past returns *)
      let past_returns = 
        if time = 0 then
          Tensor.zeros [0; n] ~kind:(Tensor.kind returns) ~device:(Tensor.device returns)
        else
          Tensor.slice returns ~dim:0 ~start:0 ~end_:time ~step:1
      in
      
      (* Get the gradients for this time *)
      let time_gradients = Tensor.slice gradients ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
      
      (* Initialize the approximation with the constant term *)
      let approximation = Tensor.clone init_vals in
      
      (* Add gradient terms for each past return *)
      for prev_time = 0 to time-1 do
        let r_prev = Tensor.slice past_returns ~dim:0 ~start:prev_time ~end_:(prev_time+1) ~step:1 in
        
        (* For each asset i *)
        for i = 0 to n-1 do
          let pi_t_i = Tensor.get approximation [|0; i|] |> Tensor.to_float0_exn in
          
          (* For each asset c in the past return *)
          for c = 0 to n-1 do
            let r_c_d = Tensor.get r_prev [|0; c|] |> Tensor.to_float0_exn in
            
            (* Get the gradient g^{ci}_{dt} *)
            let g_ci_dt = Tensor.get time_gradients [|0; c; i|] |> Tensor.to_float0_exn in
            
            (* Add the gradient term: g^{ci}_{dt} * R^c_d * I_{d≤t-1} *)
            let gradient_term = g_ci_dt *. r_c_d in
            
            (* Update the approximation for asset i *)
            Tensor.set approximation [|0; i|] (Tensor.float_vec [pi_t_i +. gradient_term])
          done
        done
      done;
      
      Tensor.copy_ time_slice ~src:approximation
    done;
    
    strategy

  (* Second-order Taylor approximation of the strategy *)
  let second_order_approximation ~initial_values ~first_derivatives ~second_derivatives ~returns =
    let n = Tensor.size returns |> List.nth 1 in  (* Number of assets *)
    let t = Tensor.size returns |> List.hd in     (* Number of time periods *)
    
    let strategy = Tensor.empty [t; n] ~kind:(Tensor.kind returns) ~device:(Tensor.device returns) in
    
    for time = 0 to t-1 do
      (* For each time point, compute the strategy based on the Taylor expansion *)
      let time_slice = Tensor.slice strategy ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
      
      (* Get the initial values for this time *)
      let init_vals = Tensor.slice initial_values ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
      
      (* Get the relevant past returns (available at this time) *)
      let past_returns = 
        if time = 0 then
          Tensor.zeros [0; n] ~kind:(Tensor.kind returns) ~device:(Tensor.device returns)
        else
          Tensor.slice returns ~dim:0 ~start:0 ~end_:time ~step:1
      in
      
      (* Initialize the approximation with the constant term *)
      let approximation = Tensor.clone init_vals in
      
      (* Add first-order terms *)
      for prev_time = 0 to time-1 do
        let r_prev = Tensor.slice past_returns ~dim:0 ~start:prev_time ~end_:(prev_time+1) ~step:1 in
        
        (* For each asset i *)
        for i = 0 to n-1 do
          let pi_t_i = Tensor.get approximation [|0; i|] |> Tensor.to_float0_exn in
          
          (* For each asset c in the past return *)
          for c = 0 to n-1 do
            let r_c_d = Tensor.get r_prev [|0; c|] |> Tensor.to_float0_exn in
            
            (* Get the first derivative g^{ci}_{dt} *)
            let g_ci_dt = Tensor.get first_derivatives [|time; i; prev_time; c|] |> Tensor.to_float0_exn in
            
            (* Add the first-order term: g^{ci}_{dt} * R^c_d * I_{d≤t-1} *)
            let first_order_term = g_ci_dt *. r_c_d in
            
            (* Update the approximation for asset i *)
            Tensor.set approximation [|0; i|] (Tensor.float_vec [pi_t_i +. first_order_term])
          done
        done
      done;
      
      (* Add second-order terms *)
      for prev_time_1 = 0 to time-1 do
        let r_prev_1 = Tensor.slice past_returns ~dim:0 ~start:prev_time_1 ~end_:(prev_time_1+1) ~step:1 in
        
        for prev_time_2 = 0 to time-1 do
          let r_prev_2 = Tensor.slice past_returns ~dim:0 ~start:prev_time_2 ~end_:(prev_time_2+1) ~step:1 in
          
          (* For each asset i *)
          for i = 0 to n-1 do
            let pi_t_i = Tensor.get approximation [|0; i|] |> Tensor.to_float0_exn in
            
            (* For each pair of assets (a,c) in the past returns *)
            for a = 0 to n-1 do
              let r_a_b = Tensor.get r_prev_1 [|0; a|] |> Tensor.to_float0_exn in
              
              for c = 0 to n-1 do
                let r_c_d = Tensor.get r_prev_2 [|0; c|] |> Tensor.to_float0_exn in
                
                (* Get the second derivative h^{aci}_{bdt} *)
                let h_aci_bdt = Tensor.get second_derivatives [|time; i; prev_time_1; a; prev_time_2; c|] 
                                |> Tensor.to_float0_exn in
                
                (* Add the second-order term: (1/2) * h^{aci}_{bdt} * R^a_b * R^c_d * I_{b,d≤t-1} *)
                let second_order_term = 0.5 *. h_aci_bdt *. r_a_b *. r_c_d in
                
                (* Update the approximation for asset i *)
                Tensor.set approximation [|0; i|] (Tensor.float_vec [pi_t_i +. second_order_term])
              done
            done
          done
        done
      done;
      
      Tensor.copy_ time_slice ~src:approximation
    done;
    
    strategy

  (* Create strategy function from coefficients *)
  let create_strategy_function ~coefficients ~t ~n ~order =
    match order with
    | 1 -> 
        (* First-order Taylor expansion *)
        let d_coef = Tensor.size coefficients |> List.hd in
        let initial_values = Tensor.slice coefficients ~dim:0 ~start:0 ~end_:t ~step:1 in
        let gradients = Tensor.slice coefficients ~dim:0 ~start:t ~end_:d_coef ~step:1 in
        let gradients = Tensor.reshape gradients [t; n; n] in
        
        (fun returns -> first_order_approximation ~initial_values ~gradients ~returns)
    | 2 ->
        (* Second-order Taylor expansion *)
        let d_coef = Tensor.size coefficients |> List.hd in
        let initial_values = Tensor.slice coefficients ~dim:0 ~start:0 ~end_:t ~step:1 in
        
        (* Extract first derivatives (sized t*n*n) *)
        let first_deriv_size = t * n * n in
        let first_derivs = Tensor.slice coefficients ~dim:0 ~start:t ~end_:(t + first_deriv_size) ~step:1 in
        let first_derivs = Tensor.reshape first_derivs [t; n; n; n] in
        
        (* Extract second derivatives (sized t*n*n*n*n) *)
        let second_derivs = Tensor.slice coefficients ~dim:0 ~start:(t + first_deriv_size) ~end_:d_coef ~step:1 in
        let second_derivs = Tensor.reshape second_derivs [t; n; n; n; n; n] in
        
        (fun returns -> second_order_approximation 
                          ~initial_values 
                          ~first_derivatives:first_derivs 
                          ~second_derivatives:second_derivs 
                          ~returns)
    | _ ->
        failwith (Printf.sprintf "Taylor expansion of order %d not implemented" order)
end

(* Robust Wasserstein Profile Inference for determining δ and α parameters *)
module RWPI = struct
  (* Find A* and lambda_0* by solving the non-robust problem *)
  let step1 ~returns =
    let n = Tensor.size returns |> List.nth 1 in
    let t = Tensor.size returns |> List.hd in
    
    (* Compute mean and covariance *)
    let mu = mean returns in
    let sigma = covariance returns in
    
    (* Regularize sigma for numerical stability *)
    let identity = Tensor.eye n in
    let reg_sigma = Tensor.add sigma (Tensor.mul_scalar identity (Scalar.float 1e-6)) in
    let inv_sigma = Tensor.inverse reg_sigma in
    
    (* Lagrangian approach to find lambda_0 *)
    let lambda_0 = ref 0.05 in  (* Initial guess *)
    let lambda_t = Array.make t 0.0 in
    
    (* Iteratively solve for A* *)
    for _ = 1 to 50 do
      (* Compute the term: λ_0 μ + Σ λ_t 1 *)
      let term = Tensor.mul_scalar mu (Scalar.float !lambda_0) in
      
      for i = 0 to t-1 do
        let one_t = Tensor.ones [n] ~kind:(Tensor.kind returns) in
        Tensor.add_ term (Tensor.mul_scalar one_t (Scalar.float lambda_t.(i)))
      done;
      
      (* Compute A* = (1/2) Σ^(-1) [λ_0 μ + Σ λ_t 1] *)
      let a_star = Tensor.matmul inv_sigma (Tensor.reshape term [-1; 1]) in
      Tensor.div_scalar_inplace a_star (Scalar.float 2.0);
      
      (* Compute expected return and sum of weights *)
      let expected_return = Tensor.matmul (Tensor.reshape mu [1; -1]) a_star |> Tensor.to_float0_exn in
      let sum_weights = Tensor.sum a_star ~dim:[0] ~keepdim:false |> Tensor.to_float0_exn in
      
      (* Update lambda values *)
      lambda_0 := !lambda_0 +. 0.01 *. (0.1 -. expected_return);  (* Target return = 0.1 *)
      for i = 0 to t-1 do
        lambda_t.(i) <- lambda_t.(i) +. 0.01 *. (1.0 -. sum_weights);
      done;
    done;
    
    (* Compute final A* *)
    let term = Tensor.mul_scalar mu (Scalar.float !lambda_0) in
    for i = 0 to t-1 do
      let one_t = Tensor.ones [n] ~kind:(Tensor.kind returns) in
      Tensor.add_ term (Tensor.mul_scalar one_t (Scalar.float lambda_t.(i)))
    done;
    
    let a_star_weights = Tensor.matmul inv_sigma (Tensor.reshape term [-1; 1]) in
    Tensor.div_scalar_inplace a_star_weights (Scalar.float 2.0);
    
    (* Normalize weights to satisfy the budget constraint *)
    let sum_weights = Tensor.sum a_star_weights ~dim:[0] ~keepdim:false in
    let normalized_weights = Tensor.div a_star_weights sum_weights in
    
    (* Construct full coefficient tensor for first-order approximation *)
    let d_coef = t + t*n in
    let a_star_full = Tensor.zeros [d_coef; n] ~kind:(Tensor.kind returns) in
    
    (* Fill in the constant terms (same allocation for all time periods) *)
    for i = 0 to t-1 do
      Tensor.copy_ 
        (Tensor.slice a_star_full ~dim:0 ~start:i ~end_:(i+1) ~step:1) 
        ~src:(Tensor.transpose normalized_weights ~dim0:0 ~dim1:1)
    done;
    
    (* Initialize gradient terms with small values *)
    let gradients = Tensor.mul_scalar (Tensor.randn [t*n; n] ~kind:(Tensor.kind returns)) (Scalar.float 0.01) in
    Tensor.copy_ (Tensor.slice a_star_full ~dim:0 ~start:t ~end_:d_coef ~step:1) ~src:gradients;
    
    a_star_full, !lambda_0

  (* Determine the Wasserstein ball radius δ *)
  let step2 ~a_star ~lambda_0 ~returns ~confidence =
    let n = Tensor.size returns |> List.nth 1 in
    let t = Tensor.size returns |> List.hd in
    
    (* Extract the constant terms from a_star *)
    let constant_terms = Tensor.slice a_star ~dim:0 ~start:0 ~end_:t ~step:1 in
    
    (* Compute the mean of constant terms across time periods *)
    let a_const = Tensor.mean constant_terms ~dim:[0] ~keepdim:false in
    
    (* Compute μ and Σ *)
    let mu = mean returns in
    let sigma = covariance returns in
    
    (* Compute μᵀΣ⁻¹μ *)
    let identity = Tensor.eye n in
    let reg_sigma = Tensor.add sigma (Tensor.mul_scalar identity (Scalar.float 1e-6)) in
    let inv_sigma = Tensor.inverse reg_sigma in
    
    let mu_sigma_mu = 
      Tensor.matmul 
        (Tensor.matmul (Tensor.reshape mu [1; -1]) inv_sigma) 
        (Tensor.reshape mu [-1; 1])
      |> Tensor.to_float0_exn
    in
    
    (* Critical value based on confidence level *)
    let chi_square_quantile =
      if confidence > 0.99 then 6.635 (* 99% *)
      else if confidence > 0.95 then 3.841 (* 95% *)
      else if confidence > 0.9 then 2.706 (* 90% *)
      else 1.642 (* 80% *)
    in
    
    (* Calculate δ according to the protocol *)
    let delta_star = chi_square_quantile /. (4.0 *. (1.0 -. mu_sigma_mu) *. float_of_int t) in
    
    delta_star

  (* Determine the worst acceptable return α_bar *)
  let step3 ~a_star ~returns ~delta ~confidence =
    let n = Tensor.size returns |> List.nth 1 in
    let t = Tensor.size returns |> List.hd in
    
    (* Extract constant terms and compute mean *)
    let constant_terms = Tensor.slice a_star ~dim:0 ~start:0 ~end_:t ~step:1 in
    let a_const = Tensor.mean constant_terms ~dim:[0] ~keepdim:false in
    
    (* Compute expected return under empirical measure *)
    let mu = mean returns in
    let expected_return = 
      Tensor.matmul (Tensor.reshape mu [1; -1]) (Tensor.reshape a_const [-1; 1]) 
      |> Tensor.to_float0_exn
    in
    
    (* Compute the norm of a_star for the penalty term *)
    let a_norm = l2_norm a_star |> Tensor.to_float0_exn in
    
    (* Compute portfolio return variance *)
    let portfolio_returns = Tensor.matmul returns (Tensor.reshape a_const [-1; 1]) in
    let port_return_var = Tensor.var portfolio_returns ~dim:[0] ~unbiased:true ~keepdim:false 
                         |> Tensor.to_float0_exn in
    
    (* Calculate quantile based on confidence level *)
    let quantile = 
      if confidence > 0.99 then 2.576
      else if confidence > 0.95 then 1.96
      else if confidence > 0.9 then 1.645
      else 1.28
    in
    
    (* Compute s0 as in the protocol *)
    let s0 = quantile *. sqrt (port_return_var /. float_of_int t) /. (sqrt delta *. a_norm) in
    
    (* Compute s0' *)
    let s0_prime = (expected_return -. (sqrt delta *. a_norm)) /. (sqrt delta *. a_norm) in
    
    (* Take the maximum of s0 and s0' *)
    let s = max s0 s0_prime in
    
    (* Compute alpha_bar *)
    let alpha_bar = expected_return -. (sqrt delta *. a_norm *. s) in
    
    alpha_bar

  (* Run the complete RWPI protocol *)
  let run ~returns ~confidence =
    (* Find A* and lambda_0* *)
    let a_star, lambda_0 = step1 ~returns in
    Printf.printf "RWPI Step 1: λ₀ = %f\n" lambda_0;
    
    (* Determine delta (Wasserstein ball radius) *)
    let delta = step2 ~a_star ~lambda_0 ~returns ~confidence in
    Printf.printf "RWPI Step 2: δ = %f\n" delta;
    
    (* Determine alpha_bar (worst acceptable return) *)
    let alpha_bar = step3 ~a_star ~returns ~delta ~confidence in
    Printf.printf "RWPI Step 3: α_bar = %f\n" alpha_bar;
    
    delta, alpha_bar, a_star
end

(* Feasible region for the robust optimization problem *)
module FeasibleRegion = struct
  (* Check if a strategy is in the feasible region *)
  let check_feasibility ~coefficients ~returns ~delta ~alpha_bar =
    let n = Tensor.size returns |> List.nth 1 in
    let t = Tensor.size returns |> List.hd in
    
    (* Convert coefficients to strategy using Taylor approximation *)
    let strategy = TaylorApproximation.create_strategy_function 
                    ~coefficients 
                    ~t 
                    ~n 
                    ~order:1 
                    returns in
    
    (* Check the budget constraint: Σ weights = 1 *)
    let budget_satisfied = ref true in
    for time = 0 to t-1 do
      let row_sum = Tensor.sum (Tensor.slice strategy ~dim:0 ~start:time ~end_:(time+1) ~step:1) 
                    ~dim:[1] ~keepdim:false |> Tensor.to_float0_exn in
      if abs_float (row_sum -. 1.0) > 1e-5 then
        budget_satisfied := false;
    done;
    
    (* Check the return constraint: EQ[A'M] - δ√||A||₂ ≥ α_bar *)
    (* Compute expected portfolio return *)
    let portfolio_returns = Tensor.matmul returns strategy in
    let expected_return = Tensor.mean portfolio_returns ~dim:[0] ~keepdim:false |> Tensor.to_float0_exn in
    
    (* Compute the norm penalty *)
    let coef_norm = l2_norm coefficients |> Tensor.to_float0_exn in
    let penalty = delta *. sqrt coef_norm in
    
    (* Worst-case expected return *)
    let worst_case_return = expected_return -. penalty in
    let return_satisfied = worst_case_return >= alpha_bar in
    
    !budget_satisfied && return_satisfied

  (* Compute the worst-case expected return *)
  let worst_case_expected_return ~coefficients ~returns ~delta =
    (* Compute expected return under empirical measure *)
    let strategy = TaylorApproximation.create_strategy_function 
                    ~coefficients 
                    ~t:(Tensor.size returns |> List.hd) 
                    ~n:(Tensor.size returns |> List.nth 1)
                    ~order:1 
                    returns in
    
    let portfolio_returns = Tensor.matmul returns strategy in
    let expected_return = Tensor.mean portfolio_returns ~dim:[0] ~keepdim:false |> Tensor.to_float0_exn in
    
    (* Compute penalty term *)
    let coef_norm = l2_norm coefficients |> Tensor.to_float0_exn in
    let penalty = delta *. sqrt coef_norm in
    
    (* Return worst-case expected return *)
    expected_return -. penalty

  (* Project coefficients onto the feasible region *)
  let project_onto_feasible_region ~coefficients ~returns ~delta ~alpha_bar =
    let n = Tensor.size returns |> List.nth 1 in
    let t = Tensor.size returns |> List.hd in
    
    (* Make a copy of the coefficients *)
    let projected = Tensor.clone coefficients in
    
    (* First, ensure the budget constraint is satisfied *)
    let constant_terms = Tensor.slice projected ~dim:0 ~start:0 ~end_:t ~step:1 in
    
    for time = 0 to t-1 do
      let time_slice = Tensor.slice constant_terms ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
      let sum = Tensor.sum time_slice ~dim:[1] ~keepdim:true in
      
      (* Normalize to sum to 1 *)
      if Tensor.to_float0_exn sum > 1e-10 then
        Tensor.div_inplace time_slice sum;
    done;
    
    (* Check if the return constraint is satisfied *)
    if check_feasibility ~coefficients:projected ~returns ~delta ~alpha_bar then
      projected
    else
      (* If not, scale down the gradient terms to reduce the penalty *)
      let gradient_terms = Tensor.slice projected ~dim:0 ~start:t ~end_:(Tensor.size projected |> List.hd) ~step:1 in
      
      (* Binary search for a scale factor that makes the projected coefficients feasible *)
      let rec find_feasible_scale scale_min scale_max iterations =
        if iterations = 0 then
          (scale_min +. scale_max) /. 2.0
        else
          let scale = (scale_min +. scale_max) /. 2.0 in
          
          (* Apply scale to gradient terms *)
          let scaled_gradient = Tensor.mul_scalar gradient_terms (Scalar.float scale) in
          Tensor.copy_ (Tensor.slice projected ~dim:0 ~start:t ~end_:(Tensor.size projected |> List.hd) ~step:1) 
                       ~src:scaled_gradient;
          
          if check_feasibility ~coefficients:projected ~returns ~delta ~alpha_bar then
            find_feasible_scale scale scale_max (iterations - 1)
          else
            find_feasible_scale scale_min scale (iterations - 1)
      in
      
      let best_scale = find_feasible_scale 0.0 1.0 10 in
      
      (* Apply the best scale *)
      let best_scaled_gradient = Tensor.mul_scalar gradient_terms (Scalar.float best_scale) in
      Tensor.copy_ (Tensor.slice projected ~dim:0 ~start:t ~end_:(Tensor.size projected |> List.hd) ~step:1) 
                  ~src:best_scaled_gradient;
      
      projected
end

(* Robust optimization solver for the dual problem *)
module RobustOptimizer = struct
  (* Dual objective function *)
  let dual_objective ~coefficients ~returns ~delta =
    (* Compute variance term: A^T Var_Q(M) A *)
    let strategy = TaylorApproximation.create_strategy_function 
                    ~coefficients 
                    ~t:(Tensor.size returns |> List.hd) 
                    ~n:(Tensor.size returns |> List.nth 1)
                    ~order:1 
                    returns in
    
    let portfolio_returns = Tensor.matmul returns strategy in
    let portfolio_variance = Tensor.var portfolio_returns ~dim:[0] ~unbiased:true ~keepdim:false 
                            |> Tensor.to_float0_exn in
    
    (* Compute penalty term: δ||A||_2 *)
    let coef_norm = l2_norm coefficients |> Tensor.to_float0_exn in
    let penalty = delta *. sqrt coef_norm in
    
    (* Compute objective: √(A^T Var_Q(M) A + δ||A||_2) *)
    sqrt (portfolio_variance +. penalty)

  (* Solve the dual optimization problem *)
  let solve_dual_problem ~returns ~delta ~alpha_bar ~config =
    let n = config.n in
    let t = config.t in
    
    Printf.printf "Solving dual problem with δ=%f, α_bar=%f...\n" delta alpha_bar;
    
    (* Initialize coefficients *)
    let d_coef = t + t*n in  (* Size for first-order approximation *)
    let coefficients = Tensor.randn [d_coef; n] ~kind:Float in
    
    (* Normalize constant terms to sum to 1 *)
    let constant_terms = Tensor.slice coefficients ~dim:0 ~start:0 ~end_:t ~step:1 in
    for time = 0 to t-1 do
      let time_slice = Tensor.slice constant_terms ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
      let sum = Tensor.sum time_slice ~dim:[1] ~keepdim:true in
      Tensor.div_inplace time_slice sum;
    done;
    
    (* Project onto the feasible region to start with a feasible point *)
    let feasible_coefficients = 
      FeasibleRegion.project_onto_feasible_region 
        ~coefficients 
        ~returns 
        ~delta 
        ~alpha_bar
    in
    
    (* Make the coefficients require gradients for optimization *)
    let trainable = Tensor.set_requires_grad feasible_coefficients ~r:true in
    
    (* Set up the optimizer *)
    let optimizer = Optimizer.adam [|trainable|] ~lr:config.learning_rate in
    
    (* Main optimization loop *)
    let best_objective = ref Float.infinity in
    let best_coefficients = ref (Tensor.clone feasible_coefficients) in
    
    for iter = 1 to config.max_iter do
      Optimizer.zero_grad optimizer;
      
      (* Compute the strategy *)
      let strategy_fn = TaylorApproximation.create_strategy_function 
                         ~coefficients:trainable 
                         ~t 
                         ~n 
                         ~order:config.taylor_order in
      
      let strategy = strategy_fn returns in
      
      (* Compute portfolio variance *)
      let portfolio_returns = Tensor.matmul returns strategy in
      let portfolio_variance = Tensor.var portfolio_returns ~dim:[0] ~unbiased:true ~keepdim:false in
      
      (* Compute the norm penalty *)
      let coef_norm = l2_norm trainable in
      let norm_penalty = Tensor.mul_scalar coef_norm (Scalar.float delta) in
      
      (* Compute objective: sqrt(A^T Var_Q(M) A + δ||A||_2) *)
      let inside_sqrt = Tensor.add portfolio_variance norm_penalty in
      let objective = Tensor.sqrt inside_sqrt in
      
      (* Compute expected return for the constraint *)
      let mean_return = Tensor.mean portfolio_returns ~dim:[0] ~keepdim:false in
      let worst_return = Tensor.sub mean_return norm_penalty in
      
      (* Add return constraint as a penalty *)
      let return_violation = Tensor.relu (Tensor.sub (Tensor.float_vec [alpha_bar]) worst_return) in
      let return_penalty = Tensor.mul_scalar return_violation (Scalar.float 1000.0) in
      
      (* Add budget constraint as a penalty *)
      let budget_penalty = ref (Tensor.zeros [] ~kind:Float) in
      for time = 0 to t-1 do
        let row_sum = Tensor.sum (Tensor.slice strategy ~dim:0 ~start:time ~end_:(time+1) ~step:1) 
                     ~dim:[1] ~keepdim:false in
        let diff = Tensor.sub row_sum (Tensor.ones_like row_sum) in
        budget_penalty := Tensor.add !budget_penalty (Tensor.mul diff diff);
      done;
      
      (* Combine objective and constraints *)
      let total_loss = 
        Tensor.add objective
          (Tensor.add return_penalty (Tensor.mul_scalar !budget_penalty (Scalar.float 1000.0)))
      in
      
      (* Backward pass and optimization step *)
      Tensor.backward total_loss;
      Optimizer.step optimizer;
      
      (* Project onto the feasible region after each step *)
      let projected = 
        FeasibleRegion.project_onto_feasible_region 
          ~coefficients:(Tensor.detach trainable) 
          ~returns 
          ~delta 
          ~alpha_bar
      in
      
      Tensor.copy_ trainable ~src:projected;
      
      (* Track the best solution *)
      let current_obj = 
        dual_objective 
          ~coefficients:(Tensor.detach trainable) 
          ~returns 
          ~delta
      in
      
      if current_obj < !best_objective then begin
        best_objective := current_obj;
        Tensor.copy_ !best_coefficients ~src:(Tensor.detach trainable);
      end;
      
      (* Print progress occasionally *)
      if iter mod 100 = 0 || iter = config.max_iter then
        Printf.printf "Iteration %d: Objective = %f\n" iter current_obj;
    done;
    
    !best_coefficients
end

(* Multi-period investment strategy *)
module MultiPeriodStrategy = struct
  (* Create a strategy *)
  let create ~n ~periods ~strategy_fn ~threshold ~transaction_cost = {
    n;
    periods;
    strategy_fn;
    threshold;
    transaction_cost;
  }

  (* Compute target investment dividing portfolio into sub-portfolios *)
  let compute_target_investment ~strategy ~returns ~time =
    let n = strategy.n in
    let t = strategy.periods in
    
    (* Allocate tensor for target investment *)
    let target = Tensor.zeros [1; n] ~kind:(Tensor.kind returns) ~device:(Tensor.device returns) in
    
    (* For each sub-portfolio *)
    for i = 1 to t do
      (* Calculate the relevant time index for this sub-portfolio *)
      let effective_time = (time + i - 1) mod t in
      
      (* Extract relevant returns for this sub-portfolio's history *)
      let relevant_returns = 
        if effective_time < t then
          (* Not enough history, use what we have *)
          let available = min effective_time t in
          if available = 0 then
            Tensor.zeros [1; n] ~kind:(Tensor.kind returns) ~device:(Tensor.device returns)
          else
            Tensor.slice returns ~dim:0 ~start:(effective_time - available) ~end_:effective_time ~step:1
        else
          (* Use the last t returns *)
          Tensor.slice returns ~dim:0 ~start:(effective_time - t) ~end_:effective_time ~step:1
      in
      
      (* Compute allocation for this sub-portfolio *)
      let sub_allocation = strategy.strategy_fn relevant_returns in
      
      (* Get the current time slice from the allocation *)
      let current_slice = 
        if Tensor.size sub_allocation |> List.hd > 0 then
          let idx = min ((Tensor.size sub_allocation |> List.hd) - 1) (effective_time mod t) in
          Tensor.slice sub_allocation ~dim:0 ~start:idx ~end_:(idx+1) ~step:1
        else
          Tensor.zeros [1; n] ~kind:(Tensor.kind returns) ~device:(Tensor.device returns)
      in
      
      (* Add to target, weighted by 1/t *)
      let weighted = Tensor.div_scalar current_slice (Scalar.float (float_of_int t)) in
      Tensor.add_ target weighted
    done;
    
    (* Ensure the weights sum to 1 *)
    let sum = Tensor.sum target ~dim:[1] ~keepdim:true in
    Tensor.div target sum
  
  (* Simulate a portfolio using the multi-period strategy with transaction costs *)
  let simulate ~strategy ~initial_wealth ~prices =
    let returns = DataUtils.prices_to_returns prices in
    let n = strategy.n in
    let t = Tensor.size returns |> List.hd in
    
    (* Initialize portfolio state *)
    let current_wealth = ref initial_wealth in
    let current_weights = ref (Tensor.ones [1; n] ~kind:Float) in
    Tensor.div_scalar_inplace !current_weights (Scalar.float (float_of_int n));
    
    (* Initialize rebalancing times *)
    let rebalance_times = ref [] in
    
    (* Initialize wealth history *)
    let wealth_history = Array.make (t+1) initial_wealth in
    wealth_history.(0) <- initial_wealth;
    
    (* Initialize transaction cost tracking *)
    let total_transaction_costs = ref 0.0 in
    
    (* Simulation loop *)
    for time = 0 to t-1 do
      (* Compute target weights for this time *)
      let target = compute_target_investment ~strategy ~returns:(Tensor.slice returns ~dim:0 ~start:0 ~end_:time ~step:1) ~time in
      
      (* Check if rebalancing is needed based on threshold *)
      let deviation = 
        let diff = Tensor.sub !current_weights target in
        let abs_diff = Tensor.abs diff in
        Tensor.max abs_diff ~dim:[1] ~keepdim:false |> fst |> Tensor.to_float0_exn
      in
      
      (* Rebalance if deviation exceeds threshold *)
      if deviation > strategy.threshold then begin
        (* Record rebalancing time *)
        rebalance_times := time :: !rebalance_times;
        
        (* Calculate transaction costs *)
        let turnover = 
          let diff = Tensor.sub target !current_weights in
          let abs_diff = Tensor.abs diff in
          Tensor.sum abs_diff ~dim:[1] ~keepdim:false |> Tensor.to_float0_exn
        in
        
        let tx_cost = turnover *. strategy.transaction_cost *. !current_wealth in
        total_transaction_costs := !total_transaction_costs +. tx_cost;
        
        (* Update wealth after transaction costs *)
        current_wealth := !current_wealth -. tx_cost;
        
        (* Update weights *)
        current_weights := target;
      end;
      
      (* Get current period returns *)
      let period_returns = Tensor.slice returns ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
      
      (* Update portfolio value *)
      let portfolio_return = 
        Tensor.sum (Tensor.mul !current_weights period_returns) ~dim:[1] ~keepdim:false
        |> Tensor.to_float0_exn
      in
      
      current_wealth := !current_wealth *. (1.0 +. portfolio_return);
      
      (* Update weights due to price changes *)
      let new_values = Tensor.mul !current_weights (Tensor.add (Tensor.ones_like period_returns) period_returns) in
      let total_value = Tensor.sum new_values ~dim:[1] ~keepdim:true in
      current_weights := Tensor.div new_values total_value;
      
      (* Record wealth *)
      wealth_history.(time+1) <- !current_wealth;
    done;
    
    (* Return simulation results *)
    {
      final_wealth = !current_wealth;
      wealth_history;
      rebalance_times = List.rev !rebalance_times;
      total_transaction_costs = !total_transaction_costs;
      final_weights = !current_weights;
    }
end

(* Robust mean-variance optimization *)
module RobustPortfolioOptimization = struct
  (* Train a multi-period robust strategy *)
  let train ~config ~historical_returns =
    Printf.printf "Training %d-period robust strategy...\n" config.t;
    
    (* Run RWPI protocol to determine delta and alpha_bar *)
    let delta, alpha_bar, _ = RWPI.run ~returns:historical_returns ~confidence:config.confidence in
    
    (* Solve the robust optimization problem *)
    let coefficients = 
      RobustOptimizer.solve_dual_problem
        ~returns:historical_returns
        ~delta
        ~alpha_bar
        ~config
    in
    
    (* Create a strategy function from the optimal coefficients *)
    let strategy_fn = 
      TaylorApproximation.create_strategy_function
        ~coefficients
        ~t:config.t
        ~n:config.n
        ~order:config.taylor_order
    in
    
    (* Create the multi-period investment strategy *)
    let strategy = 
      MultiPeriodStrategy.create
        ~n:config.n
        ~periods:config.t
        ~strategy_fn
        ~threshold:config.rebalance_threshold
        ~transaction_cost:config.transaction_cost
    in
    
    strategy, delta, alpha_bar

  (* Train and evaluate a strategy *)
  let train_and_evaluate ~config ~train_data ~test_data ~initial_wealth =
    (* Convert price data to returns for training *)
    let train_returns = DataUtils.prices_to_returns train_data in
    
    (* Train the strategy *)
    let strategy, delta, alpha_bar = train ~config ~historical_returns:train_returns in
    
    (* Simulate the strategy on test data *)
    let simulation_result = MultiPeriodStrategy.simulate ~strategy ~initial_wealth ~prices:test_data in
    
    (* Calculate performance metrics *)
    let performance = Performance.calculate_metrics ~simulation_result ~initial_wealth in
    
    (* Print performance summary *)
    Performance.print_summary 
      ~name:(Printf.sprintf "%d-Period Robust Strategy" config.t)
      ~performance;
    
    strategy, performance, delta, alpha_bar

  (* Compare multiple strategies *)
  let compare_strategies ~strategies ~test_data ~initial_wealth =
    let results = ref [] in
    
    List.iter (fun (name, strategy) ->
      Printf.printf "\nEvaluating strategy: %s\n" name;
      
      (* Simulate the strategy *)
      let simulation_result = MultiPeriodStrategy.simulate ~strategy ~initial_wealth ~prices:test_data in
      
      (* Calculate performance *)
      let performance = Performance.calculate_metrics ~simulation_result ~initial_wealth in
      
      (* Print summary *)
      Performance.print_summary ~name ~performance;
      
      (* Store results *)
      results := (name, performance) :: !results
    ) strategies;
    
    List.rev !results
end

(* Analysis of the effect of number of periods *)
module PeriodAnalysis = struct
  (* Simulate with different numbers of periods *)
  let simulate_with_different_periods ~train_data ~test_data ~initial_wealth ~period_counts =
    let n = Tensor.size train_data |> List.nth 1 in
    let train_returns = DataUtils.prices_to_returns train_data in
    
    (* Results for each period count *)
    let results = ref [] in
    
    List.iter (fun periods ->
      Printf.printf "\n=== Testing %d-Period Strategy ===\n" periods;
      
      (* Create configuration *)
      let config = {
        n;
        t = periods;
        confidence = 0.95;
        learning_rate = 0.01;
        max_iter = 1000;
        rebalance_threshold = 0.05;
        transaction_cost = 0.002;
        taylor_order = 1;
      } in
      
      (* Train and evaluate the strategy *)
      let strategy, performance, delta, alpha_bar = 
        RobustPortfolioOptimization.train_and_evaluate
          ~config
          ~train_data
          ~test_data
          ~initial_wealth
      in
      
      (* Store results *)
      results := (periods, performance, delta, alpha_bar) :: !results
    ) period_counts;
    
    List.rev !results
    
  (* Split portfolio into n sub-portfolios starting at different times *)
  let create_split_portfolios ~strategy ~t ~n ~threshold ~transaction_cost =
    let sub_portfolios = ref [] in
    
    (* Create t sub-portfolios *)
    for i = 0 to t-1 do
      let modified_strategy_fn returns =
        (* Get the base allocation from the original strategy *)
        let base_allocation = strategy returns in
        
        (* The size of the result *)
        let time_periods = Tensor.size base_allocation |> List.hd in
        let num_assets = Tensor.size base_allocation |> List.nth 1 in
        
        (* Create result tensor initialized with zeros *)
        let result = Tensor.zeros [time_periods; num_assets] ~kind:(Tensor.kind base_allocation) in
        
        (* Only copy values for time points that match our offset *)
        for t = i to time_periods-1 by t do
          if t < time_periods then
            let source = Tensor.slice base_allocation ~dim:0 ~start:t ~end_:(t+1) ~step:1 in
            Tensor.copy_ (Tensor.slice result ~dim:0 ~start:t ~end_:(t+1) ~step:1) ~src:source
          else
            (* Offset is beyond available data, keep zeros *)
            ()
        done;
        
        result
      in
      
      (* Create the sub-portfolio *)
      let sub_portfolio = 
        MultiPeriodStrategy.create
          ~n
          ~periods:t
          ~strategy_fn:modified_strategy_fn
          ~threshold
          ~transaction_cost
      in
      
      sub_portfolios := sub_portfolio :: !sub_portfolios
    done;
    
    List.rev !sub_portfolios
    
  (* Combine sub-portfolios to create a final multi-period strategy *)
  let combine_sub_portfolios ~sub_portfolios ~n ~t ~threshold ~transaction_cost =
    (* Create a combined strategy function *)
    let combined_strategy_fn returns =
      (* Get allocations from all sub-portfolios *)
      let sub_allocations = 
        List.map (fun strategy -> strategy.MultiPeriodStrategy.strategy_fn returns) sub_portfolios
      in
      
      (* The size of the result *)
      let time_periods = 
        match sub_allocations with
        | [] -> 0
        | first :: _ -> Tensor.size first |> List.hd
      in
      
      (* Create result tensor *)
      let result = Tensor.zeros [time_periods; n] ~kind:Float in
      
      (* For each time period *)
      for time = 0 to time_periods-1 do
        (* For each sub-portfolio *)
        List.iteri (fun _ sub_allocation ->
          (* Extract allocation for this time *)
          let alloc = Tensor.slice sub_allocation ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
          
          (* Scale by 1/t and add to result *)
          let scaled = Tensor.div_scalar alloc (Scalar.float (float_of_int t)) in
          Tensor.add_ (Tensor.slice result ~dim:0 ~start:time ~end_:(time+1) ~step:1) scaled
        ) sub_allocations
      done;
      
      (* Ensure the result sums to 1 for each time period *)
      for time = 0 to time_periods-1 do
        let time_slice = Tensor.slice result ~dim:0 ~start:time ~end_:(time+1) ~step:1 in
        let sum = Tensor.sum time_slice ~dim:[1] ~keepdim:true in
        
        (* Only normalize if sum is not close to zero *)
        if Tensor.to_float0_exn sum > 1e-6 then
          Tensor.div_inplace time_slice sum
      done;
      
      result
    in
    
    MultiPeriodStrategy.create
      ~n
      ~periods:t
      ~strategy_fn:combined_strategy_fn
      ~threshold
      ~transaction_cost
end