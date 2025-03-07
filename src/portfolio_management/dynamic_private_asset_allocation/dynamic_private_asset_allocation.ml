open Torch

(* Time condition states *)
module TimeConditions = struct
  type state = Pos | Neg
  
  (* Convert state to integer representation *)
  let to_int = function
    | Pos -> 1
    | Neg -> 2
  
  (* Convert integer to state *)
  let of_int = function
    | 1 -> Pos
    | 2 -> Neg
    | _ -> failwith "Invalid time condition state"
    
  (* Transition probabilities *)
  type transition_matrix = {
    p11: float; (* Pos to Pos *)
    p12: float; (* Pos to Neg *)
    p21: float; (* Neg to Pos *)
    p22: float; (* Neg to Neg *)
  }
  
  (* Get transition probability from state s to state s' *)
  let transition_prob matrix s s' =
    match s, s' with
    | Pos, Pos -> matrix.p11
    | Pos, Neg -> matrix.p12
    | Neg, Pos -> matrix.p21
    | Neg, Neg -> matrix.p22
  
  (* Create a valid transition matrix (rows sum to 1) *)
  let create_transition_matrix p11 p21 =
    let p12 = 1.0 -. p11 in
    let p22 = 1.0 -. p21 in
    { p11; p12; p21; p22 }
    
  (* Sample next state given current state and transition matrix *)
  let sample_next_state matrix current_state =
    let u = Random.float 1.0 in
    match current_state with
    | Pos -> if u < matrix.p11 then Pos else Neg
    | Neg -> if u < matrix.p21 then Pos else Neg
end

(* Asset returns *)
module AssetReturns = struct
  (* Parameters for asset returns that depend on time condition *)
  type return_parameters = {
    mu_s: float;        (* Expected stock return *)
    sigma_s: float;     (* Stock volatility *)
    rf: float;          (* Risk-free rate *)
    sigma_p: float;     (* PE volatility *)
    rho: float;         (* Correlation between PE and stock returns *)
    nu_p: float;        (* Time condition component of PE returns *)
  }
  
  (* Return parameters conditional on time condition state *)
  type state_dependent_returns = {
    pos: return_parameters;
    neg: return_parameters;
  }
  
  (* Get parameters for the current state *)
  let get_parameters params state =
    match state with
    | TimeConditions.Pos -> params.pos
    | TimeConditions.Neg -> params.neg
  
  (* PE return process parameters - implementing Getmansky, Lo, and Makarov (2004) model *)
  type pe_process_params = {
    rho_p1: float;         (* First autocorrelation parameter *)
    rho_p2: float;         (* Second autocorrelation parameter *)
    theta0: float;         (* Current period weight in smoothing *)
    theta1: float;         (* One-period lag weight in smoothing *)
    theta2: float;         (* Two-period lag weight in smoothing *)
  }
  
  (* Default PE process parameters based on typical values from the literature *)
  let default_pe_process_params () = {
    rho_p1 = 0.6;          (* First-order autocorrelation *)
    rho_p2 = 0.2;          (* Second-order autocorrelation *)
    theta0 = 0.7;          (* Current period has highest weight *)
    theta1 = 0.2;          (* One-period lag has moderate weight *)
    theta2 = 0.1;          (* Two-period lag has lowest weight *)
  }
  
  (* Validate smoothing weights - they should sum to 1 *)
  let validate_smoothing_weights params =
    let sum = params.theta0 +. params.theta1 +. params.theta2 in
    if abs_float (sum -. 1.0) > 1e-6 then
      let scale = 1.0 /. sum in
      {
        params with
        theta0 = params.theta0 *. scale;
        theta1 = params.theta1 *. scale;
        theta2 = params.theta2 *. scale;
      }
    else
      params
  
  (* Generate true (unobserved) PE return *)
  let generate_true_pe_return mu_p sigma_p eps_p =
    exp (mu_p +. sigma_p *. eps_p)
  
  (* Apply return smoothing following Getmansky, Lo, and Makarov (2004) *)
  let apply_return_smoothing params r_true r_prev1 r_prev2 =
    let log_r_true = log r_true in
    let log_r_prev1 = log r_prev1 in
    let log_r_prev2 = log r_prev2 in
    
    (* Calculate smoothed log return *)
    let log_r_smoothed = 
      params.theta0 *. log_r_true +.
      params.theta1 *. log_r_prev1 +.
      params.theta2 *. log_r_prev2
    in
    
    (* Convert back to gross return *)
    exp log_r_smoothed
  
  (* Calculate observed PE return given the true economic return and lagged values *)
  let calculate_observed_pe_return params true_return prev_true1 prev_true2 =
    let validated_params = validate_smoothing_weights params in
    apply_return_smoothing validated_params true_return prev_true1 prev_true2
  
  (* Update expected PE return *)
  let update_expected_pe_return params prev_mu_p log_rp_prev nu_p =
    params.rho_p1 *. prev_mu_p +. params.rho_p2 *. log_rp_prev +. nu_p
  
  (* Calculate true PE return given the expected return and volatility *)
  let calculate_true_pe_return mu_p sigma_p =
    let eps_p = Random.float_gaussian () in
    generate_true_pe_return mu_p sigma_p eps_p, eps_p
  
  (* Cache for previous true PE returns *)
  type pe_return_history = {
    true_returns: float array;
    observed_returns: float array;
    current_idx: int;
  }
  
  (* Initialize return history with starting values *)
  let init_pe_return_history initial_return size =
    let true_returns = Array.make size initial_return in
    let observed_returns = Array.make size initial_return in
    { true_returns; observed_returns; current_idx = 0 }
  
  (* Update return history with new values *)
  let update_pe_return_history history true_return observed_return =
    let new_idx = (history.current_idx + 1) mod Array.length history.true_returns in
    history.true_returns.(new_idx) <- true_return;
    history.observed_returns.(new_idx) <- observed_return;
    { history with current_idx = new_idx }
  
  (* Get previous return values from history *)
  let get_previous_returns history lag =
    let idx = (history.current_idx - lag + Array.length history.true_returns) mod Array.length history.true_returns in
    (history.true_returns.(idx), history.observed_returns.(idx))
  
  (* Generate log returns for PE and stocks with proper return smoothing *)
  let generate_returns_with_history rparams pe_params mu_p_prev history state =
    let params = get_parameters rparams state in
    
    (* Get previous returns for smoothing *)
    let (prev_true1, prev_obs1) = get_previous_returns history 1 in
    let (prev_true2, prev_obs2) = get_previous_returns history 2 in
    
    (* Update expected PE return *)
    let log_prev_obs = log prev_obs1 in
    let mu_p = update_expected_pe_return pe_params mu_p_prev log_prev_obs params.nu_p in
    
    (* Generate correlated normal random variables *)
    let z1 = Random.float_gaussian () in
    let z2 = Random.float_gaussian () in
    
    (* Apply correlation *)
    let eps_p = z1 in
    let eps_s = params.rho *. z1 +. sqrt (1.0 -. params.rho *. params.rho) *. z2 in
    
    (* Calculate true PE return *)
    let true_rp = generate_true_pe_return mu_p params.sigma_p eps_p in
    
    (* Apply return smoothing *)
    let rp = calculate_observed_pe_return pe_params true_rp prev_true1 prev_true2 in
    
    (* Calculate stock return *)
    let log_rs = params.mu_s +. params.sigma_s *. eps_s in
    let rs = exp log_rs in
    
    (* Risk-free return *)
    let rf = 1.0 +. params.rf in
    
    (* Update history *)
    let new_history = update_pe_return_history history true_rp rp in
    
    (* Return results *)
    (rp, rs, rf, mu_p, log rp, new_history)
  
  (* Generate log returns for PE and stocks - simplified version without full history tracking *)
  let generate_returns rparams pe_params mu_p_prev log_rp_prev state =
    let params = get_parameters rparams state in
    
    (* Update expected PE return *)
    let mu_p = update_expected_pe_return pe_params mu_p_prev log_rp_prev params.nu_p in
    
    (* Generate correlated normal random variables *)
    let z1 = Random.float_gaussian () in
    let z2 = Random.float_gaussian () in
    
    (* Apply correlation *)
    let eps_p = z1 in
    let eps_s = params.rho *. z1 +. sqrt (1.0 -. params.rho *. params.rho) *. z2 in
    
    (* Calculate log returns *)
    let log_rp = mu_p +. params.sigma_p *. eps_p in
    let log_rs = params.mu_s +. params.sigma_s *. eps_s in
    
    (* Return gross returns and updated expected PE return *)
    let rp = exp log_rp in
    let rs = exp log_rs in
    let rf = 1.0 +. params.rf in
    
    (rp, rs, rf, mu_p, log_rp)
  
  (* Simple function to calculate return statistics *)
  let calculate_return_statistics returns =
    let n = Array.length returns in
    if n = 0 then (0.0, 0.0, 0.0)
    else
      (* Calculate mean *)
      let sum = Array.fold_left (+.) 0.0 returns in
      let mean = sum /. float_of_int n in
      
      (* Calculate variance *)
      let sum_sq_dev = 
        Array.fold_left (fun acc r -> acc +. ((r -. mean) ** 2.0)) 0.0 returns 
      in
      let variance = sum_sq_dev /. float_of_int (n - 1) in
      let std_dev = sqrt variance in
      
      (* Calculate first-order autocorrelation *)
      let sum_prod = ref 0.0 in
      let sum_sq = ref 0.0 in
      for i = 0 to n - 2 do
        sum_prod := !sum_prod +. (returns.(i) -. mean) *. (returns.(i + 1) -. mean);
        sum_sq := !sum_sq +. ((returns.(i) -. mean) ** 2.0);
      done;
      
      let autocorr = 
        if !sum_sq > 0.0 then !sum_prod /. !sum_sq
        else 0.0
      in
      
      (mean, std_dev, autocorr)
  
  (* Calculate covariance between two return series *)
  let calculate_covariance returns1 returns2 =
    let n = min (Array.length returns1) (Array.length returns2) in
    if n < 2 then 0.0
    else
      (* Calculate means *)
      let sum1 = Array.fold_left (+.) 0.0 (Array.sub returns1 0 n) in
      let sum2 = Array.fold_left (+.) 0.0 (Array.sub returns2 0 n) in
      let mean1 = sum1 /. float_of_int n in
      let mean2 = sum2 /. float_of_int n in
      
      (* Calculate covariance *)
      let sum_prod = ref 0.0 in
      for i = 0 to n - 1 do
        sum_prod := !sum_prod +. (returns1.(i) -. mean1) *. (returns2.(i) -. mean2);
      done;
      
      !sum_prod /. float_of_int (n - 1)
  
  (* Calculate correlation between two return series *)
  let calculate_correlation returns1 returns2 =
    let n = min (Array.length returns1) (Array.length returns2) in
    if n < 2 then 0.0
    else
      let (_, std1, _) = calculate_return_statistics (Array.sub returns1 0 n) in
      let (_, std2, _) = calculate_return_statistics (Array.sub returns2 0 n) in
      
      if std1 > 0.0 && std2 > 0.0 then
        calculate_covariance returns1 returns2 /. (std1 *. std2)
      else
        0.0
        
  (* Unsmooth returns following Getmansky, Lo, and Makarov (2004) *)
  let unsmooth_returns pe_params observed_returns =
    (* Ensure we have enough data *)
    let n = Array.length observed_returns in
    if n < 3 then observed_returns
    else
      (* Validate smoothing weights *)
      let validated_params = validate_smoothing_weights pe_params in
      
      (* Convert to log returns *)
      let log_returns = Array.map log observed_returns in
      
      (* Create array for unsmoothed returns *)
      let unsmoothed_log = Array.make n 0.0 in
      
      (* Apply unsmoothing formula for each return except first two *)
      for i = 2 to n - 1 do
        let smoothed = log_returns.(i) in
        let prev1 = log_returns.(i-1) in
        let prev2 = log_returns.(i-2) in
        
        (* Solve for the true return *)
        unsmoothed_log.(i) <- 
          (smoothed -. validated_params.theta1 *. prev1 -. validated_params.theta2 *. prev2) /.
          validated_params.theta0;
      done;
      
      (* Initial values are kept as is *)
      unsmoothed_log.(0) <- log_returns.(0);
      unsmoothed_log.(1) <- log_returns.(1);
      
      (* Convert back to gross returns *)
      Array.map exp unsmoothed_log
end

(* Capital calls and distributions module *)
module CapitalFlows = struct
  (* Parameters for capital calls and distributions depending on time condition *)
  type flow_parameters = {
    lambda_k: float;  (* Fraction of uncalled commitments called *)
    lambda_n: float;  (* Fraction of new commitments called *)
    lambda_d: float;  (* Distribution rate *)
    alpha: float;     (* Discount on PE liquidation in default *)
  }
  
  (* State-dependent flow parameters *)
  type state_dependent_flows = {
    pos: flow_parameters;
    neg: flow_parameters;
  }
  
  (* Get parameters for current state *)
  let get_parameters params state =
    match state with
    | TimeConditions.Pos -> params.pos
    | TimeConditions.Neg -> params.neg
end

(* Risk budget module *)
module RiskBudget = struct
  (* Risk weight parameters *)
  type risk_weights = {
    theta_b: float;   (* Risk weight for bonds *)
    theta_s: float;   (* Risk weight for stocks *)
    theta_p: float;   (* Risk weight for PE *)
    kappa: float;     (* Cost parameter for violating risk budget *)
    threshold: float; (* Risk threshold *)
  }
  
  (* Calculate portfolio risk weight *)
  let portfolio_risk_weight weights b s stock_adj_cost p w total_wealth =
    if total_wealth <= 0.0 then 0.0
    else
      let numerator = 
        weights.theta_b *. b +. 
        weights.theta_s *. (s +. stock_adj_cost) +.
        weights.theta_p *. p 
      in
      let denominator = b +. s +. stock_adj_cost +. p in
      
      if denominator <= 0.0 then 0.0
      else numerator /. denominator
  
  (* Calculate portfolio risk weight in default *)
  let default_risk_weight weights b s stock_adj_cost p w total_wealth =
    if total_wealth <= 0.0 then 0.0
    else
      let numerator = 
        weights.theta_b *. b +. 
        weights.theta_s *. (s +. stock_adj_cost)
        (* No PE weight in default case *)
      in
      let denominator = b +. s +. stock_adj_cost +. p in
      
      if denominator <= 0.0 then 0.0
      else numerator /. denominator
  
  (* Calculate risk cost *)
  let risk_cost weights risk_weight total_wealth =
    if risk_weight <= weights.threshold then 0.0
    else
      let excess = risk_weight -. weights.threshold in
      weights.kappa *. excess *. excess *. total_wealth
      
  (* Calculate marginal risk cost - derivative *)
  let marginal_risk_cost weights risk_weight total_wealth asset_type =
    if risk_weight <= weights.threshold then 0.0
    else
      let excess = risk_weight -. weights.threshold in
      let asset_weight = match asset_type with
        | `Bond -> weights.theta_b
        | `Stock -> weights.theta_s
        | `PE -> weights.theta_p
      in
      2.0 *. weights.kappa *. excess *. asset_weight *. total_wealth
      
  (* Calculate risk-adjusted returns - return minus risk cost *)
  let risk_adjusted_return weights expected_return risk_weight total_wealth asset_type =
    let marginal_cost = marginal_risk_cost weights risk_weight total_wealth asset_type in
    expected_return -. marginal_cost
    
  (* Check if portfolio is within risk budget *)
  let is_within_budget weights risk_weight =
    risk_weight <= weights.threshold
    
  (* Calculate how much of an asset can be added before breaching the risk budget *)
  let max_allocation_within_budget weights current_b current_s current_p total_wealth asset_type =
    let current_risk = 
      portfolio_risk_weight 
        weights 
        current_b 
        current_s 
        0.0  (* Ignoring adjustment costs for simplicity *)
        current_p 
        (current_b +. current_s +. current_p)  (* Current liquid wealth *)
        total_wealth
    in
    
    (* If already at or above threshold, can't add any more risky assets *)
    if current_risk >= weights.threshold then
      0.0
    else
      match asset_type with
      | `Bond -> 
          (* Bonds have zero risk weight, can add unlimited amount *)
          Float.infinity
      | `Stock ->
          (* Calculate how much stock can be added before hitting threshold *)
          let denominator = weights.theta_s -. weights.threshold in
          if abs_float denominator < 1e-6 then
            Float.infinity  (* If stock weight equals threshold, can add unlimited amount *)
          else if denominator < 0.0 then
            Float.infinity  (* If stock weight is below threshold *)
          else
            let current_assets = current_b +. current_s +. current_p in
            let numerator = 
              weights.threshold *. current_assets -. 
              (weights.theta_b *. current_b +. weights.theta_s *. current_s +. weights.theta_p *. current_p)
            in
            max 0.0 (numerator /. denominator)
      | `PE ->
          (* Calculate how much PE can be added before hitting threshold *)
          let denominator = weights.theta_p -. weights.threshold in
          if abs_float denominator < 1e-6 then
            Float.infinity
          else if denominator < 0.0 then
            Float.infinity
          else
            let current_assets = current_b +. current_s +. current_p in
            let numerator = 
              weights.threshold *. current_assets -. 
              (weights.theta_b *. current_b +. weights.theta_s *. current_s +. weights.theta_p *. current_p)
            in
            max 0.0 (numerator /. denominator)
end

(* Dynamic model parameter module *)
module ModelParams = struct
  (* Full set of model parameters *)
  type t = {
    business_cycle: TimeConditions.transition_matrix;
    asset_returns: AssetReturns.state_dependent_returns;
    pe_process: AssetReturns.pe_process_params;
    capital_flows: CapitalFlows.state_dependent_flows;
    risk_budget: RiskBudget.risk_weights;
    gamma: float;                (* Risk aversion coefficient *)
    epsilon_n: float;            (* PE commitment adjustment cost parameter *)
    epsilon_s: float;            (* Stock adjustment cost parameter *)
    n_bar: float;                (* Target PE commitment rate *)
    s_bar: float;                (* Target stock allocation *)
    terminal_time: int;          (* Terminal time T *)
    time_step: float;            (* Time step in years (e.g., 0.25 for quarterly) *)
    use_smoothed_returns: bool;  (* Whether to use smoothed returns (true) or unsmoothed (false) *)
  }
  
  (* Create default model parameters calibrated to data *)
  let default () = {
    (* Time condition transition probabilities *)
    business_cycle = TimeConditions.create_transition_matrix 0.8 0.2;
    
    (* Asset return parameters by time condition state *)
    asset_returns = {
      pos = {
        mu_s = 0.01;       (* 1% quarterly expected stock return in pos *)
        sigma_s = 0.10;     (* 10% quarterly stock volatility in pos *)
        rf = 0.005;         (* 0.5% quarterly risk-free rate in pos *)
        sigma_p = 0.12;     (* 12% quarterly PE volatility in pos *)
        rho = 0.6;          (* 0.6 correlation between PE and stocks in pos *)
        nu_p = 0.01;        (* 1% quarterly PE return time condition component in pos *)
      };
      neg = {
        mu_s = 0.025;       (* 2.5% quarterly expected stock return in neg *)
        sigma_s = 0.08;     (* 8% quarterly stock volatility in neg *)
        rf = 0.01;          (* 1% quarterly risk-free rate in neg *)
        sigma_p = 0.10;     (* 10% quarterly PE volatility in neg *)
        rho = 0.7;          (* 0.7 correlation between PE and stocks in neg *)
        nu_p = 0.03;        (* 3% quarterly PE return time condition component in neg *)
      };
    };
    
    (* PE return process parameters - following Getmansky, Lo, and Makarov (2004) *)
    pe_process = {
      rho_p1 = 0.6;         (* First-order autocorrelation *)
      rho_p2 = 0.2;         (* Second-order autocorrelation *)
      theta0 = 0.7;         (* Current period weight in smoothing *)
      theta1 = 0.2;         (* One-period lag weight in smoothing *)
      theta2 = 0.1;         (* Two-period lag weight in smoothing *)
    };
    
    (* Capital flow parameters by time condition state *)
    capital_flows = {
      pos = {
        lambda_k = 0.05;     (* 5% of uncalled commitments called per quarter in pos *)
        lambda_n = 0.05;     (* 5% of new commitments called immediately in pos *)
        lambda_d = 0.02;     (* 2% distribution rate per quarter in pos *)
        alpha = 0.7;         (* 30% discount on PE liquidation in pos *)
      };
      neg = {
        lambda_k = 0.10;     (* 10% of uncalled commitments called per quarter in neg *)
        lambda_n = 0.10;     (* 10% of new commitments called immediately in neg *)
        lambda_d = 0.04;     (* 4% distribution rate per quarter in neg *)
        alpha = 0.8;         (* 20% discount on PE liquidation in neg *)
      };
    };
    
    (* Risk budget parameters - 50% risk charge for both stocks and PE *)
    risk_budget = {
      theta_b = 0.0;     (* Bonds have zero risk *)
      theta_s = 0.5;     (* 50% risk charge for stocks *)
      theta_p = 0.5;     (* 50% risk charge for PE *)
      kappa = 5.0;       (* Penalty parameter *)
      threshold = 1.0;   (* Risk threshold *)
    };
    
    gamma = 5.0;         (* Risk aversion coefficient *)
    epsilon_n = 0.0001;  (* PE adjustment cost *)
    epsilon_s = 0.0001;  (* Stock adjustment cost *)
    n_bar = 0.05;        (* Target PE commitment rate of 5% *)
    s_bar = 0.5;         (* Target stock allocation of 50% *)
    terminal_time = 40;  (* 10 years with quarterly time steps *)
    time_step = 0.25;    (* Quarterly time steps (0.25 years) *)
    use_smoothed_returns = true; (* Use smoothed returns by default *)
  }
  
  (* Create a more aggressive parameter set with higher PE allocation *)
  let aggressive () =
    let default_params = default () in
    {
      default_params with
      n_bar = 0.10;        (* Target PE commitment rate of 10% *)
      s_bar = 0.60;        (* Target stock allocation of 60% *)
      gamma = 3.0;         (* Lower risk aversion *)
    }
  
  (* Create a more conservative parameter set with lower PE allocation *)
  let conservative () =
    let default_params = default () in
    {
      default_params with
      n_bar = 0.03;        (* Target PE commitment rate of 3% *)
      s_bar = 0.40;        (* Target stock allocation of 40% *)
      gamma = 7.0;         (* Higher risk aversion *)
    }
  
  (* Create a parameter set that ignores time condition variation *)
  let naive () =
    let default_params = default () in
    
    (* Average the pos and neg parameters *)
    let avg_asset_returns = {
      AssetReturns.mu_s = (default_params.asset_returns.pos.mu_s +. 
                           default_params.asset_returns.neg.mu_s) /. 2.0;
      sigma_s = (default_params.asset_returns.pos.sigma_s +. 
                 default_params.asset_returns.neg.sigma_s) /. 2.0;
      rf = (default_params.asset_returns.pos.rf +. 
            default_params.asset_returns.neg.rf) /. 2.0;
      sigma_p = (default_params.asset_returns.pos.sigma_p +. 
                 default_params.asset_returns.neg.sigma_p) /. 2.0;
      rho = (default_params.asset_returns.pos.rho +. 
             default_params.asset_returns.neg.rho) /. 2.0;
      nu_p = (default_params.asset_returns.pos.nu_p +. 
              default_params.asset_returns.neg.nu_p) /. 2.0;
    } in
    
    let avg_capital_flows = {
      CapitalFlows.lambda_k = (default_params.capital_flows.pos.lambda_k +.
                              default_params.capital_flows.neg.lambda_k) /. 2.0;
      lambda_n = (default_params.capital_flows.pos.lambda_n +.
                 default_params.capital_flows.neg.lambda_n) /. 2.0;
      lambda_d = (default_params.capital_flows.pos.lambda_d +.
                 default_params.capital_flows.neg.lambda_d) /. 2.0;
      alpha = (default_params.capital_flows.pos.alpha +.
              default_params.capital_flows.neg.alpha) /. 2.0;
    } in
    
    (* Create parameters with the same values for both states *)
    {
      default_params with
      business_cycle = TimeConditions.create_transition_matrix 0.5 0.5;  (* Equal probability of each state *)
      asset_returns = {
        pos = avg_asset_returns;
        neg = avg_asset_returns;
      };
      capital_flows = {
        pos = avg_capital_flows;
        neg = avg_capital_flows;
      };
    }
    
  (* Create parameters for a scenario with higher risk charges *)
  let high_risk_charge () =
    let default_params = default () in
    {
      default_params with
      risk_budget = {
        default_params.risk_budget with
        theta_s = 1.0;  (* 100% risk charge for stocks *)
        theta_p = 1.0;  (* 100% risk charge for PE *)
      }
    }
end

(* Gaussian Process for value function approximation *)
module GP = struct
  (* Matrix operations for GP computations *)
  module Matrix = struct
    (* Simple matrix type *)
    type t = float array array
    
    (* Create a matrix of given dimensions *)
    let create n m init_value =
      Array.init n (fun _ -> Array.make m init_value)
    
    (* Matrix addition *)
    let add a b =
      let n = Array.length a in
      let m = Array.length a.(0) in
      let result = create n m 0.0 in
      for i = 0 to n - 1 do
        for j = 0 to m - 1 do
          result.(i).(j) <- a.(i).(j) +. b.(i).(j)
        done
      done;
      result
    
    (* Matrix multiplication *)
    let mul a b =
      let n = Array.length a in
      let m = Array.length b.(0) in
      let p = Array.length a.(0) in
      let result = create n m 0.0 in
      for i = 0 to n - 1 do
        for j = 0 to m - 1 do
          for k = 0 to p - 1 do
            result.(i).(j) <- result.(i).(j) +. a.(i).(k) *. b.(k).(j)
          done
        done
      done;
      result
    
    (* Matrix-vector multiplication *)
    let mul_vec m v =
      let n = Array.length m in
      let result = Array.make n 0.0 in
      for i = 0 to n - 1 do
        for j = 0 to Array.length v - 1 do
          result.(i) <- result.(i) +. m.(i).(j) *. v.(j)
        done
      done;
      result
    
    (* Transpose a matrix *)
    let transpose m =
      let n = Array.length m in
      let m_cols = if n > 0 then Array.length m.(0) else 0 in
      let result = create m_cols n 0.0 in
      for i = 0 to n - 1 do
        for j = 0 to m_cols - 1 do
          result.(j).(i) <- m.(i).(j)
        done
      done;
      result
    
    (* Cholesky decomposition for positive definite matrices *)
    let cholesky m =
      let n = Array.length m in
      let l = create n n 0.0 in
      
      for i = 0 to n - 1 do
        for j = 0 to i do
          let s = ref 0.0 in
          for k = 0 to j - 1 do
            s := !s +. l.(i).(k) *. l.(j).(k)
          done;
          
          if i = j then
            l.(i).(j) <- sqrt (m.(i).(i) -. !s)
          else
            l.(i).(j) <- (1.0 /. l.(j).(j)) *. (m.(i).(j) -. !s)
        done
      done;
      
      l
    
    (* Solve Ax = b using Cholesky decomposition *)
    let solve_cholesky l b =
      let n = Array.length l in
      let y = Array.make n 0.0 in
      let x = Array.make n 0.0 in
      
      (* Solve Ly = b *)
      for i = 0 to n - 1 do
        let s = ref 0.0 in
        for j = 0 to i - 1 do
          s := !s +. l.(i).(j) *. y.(j)
        done;
        y.(i) <- (b.(i) -. !s) /. l.(i).(i)
      done;
      
      (* Solve L^T x = y *)
      for i = n - 1 downto 0 do
        let s = ref 0.0 in
        for j = i + 1 to n - 1 do
          s := !s +. l.(j).(i) *. x.(j)
        done;
        x.(i) <- (y.(i) -. !s) /. l.(i).(i)
      done;
      
      x
    
    (* Compute log determinant using Cholesky decomposition *)
    let log_det_cholesky l =
      let n = Array.length l in
      let result = ref 0.0 in
      for i = 0 to n - 1 do
        result := !result +. 2.0 *. log l.(i).(i)
      done;
      !result
    
    (* Add noise to diagonal *)
    let add_diagonal m value =
      let n = Array.length m in
      let result = Array.make_matrix n n 0.0 in
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          result.(i).(j) <- m.(i).(j)
        done;
        result.(i).(i) <- result.(i).(i) +. value
      done;
      result
  end
  
  (* Matern 5/2 kernel *)
  let matern52_kernel x x' lengthscales sigma_f =
    let sq_dist = 
      Array.mapi (fun i xi -> 
        let x'i = x'.(i) in
        let di = lengthscales.(i) in
        ((xi -. x'i) /. di) ** 2.0
      ) x
      |> Array.fold_left (+.) 0.0
    in
    let r = sqrt sq_dist in
    let term1 = 1.0 +. sqrt 5.0 *. r +. (5.0 /. 3.0) *. sq_dist in
    let term2 = exp (-. sqrt 5.0 *. r) in
    sigma_f ** 2.0 *. term1 *. term2
  
  (* Gaussian Process parameters *)
  type gp_params = {
    mean: float;               (* Prior mean m(x) *)
    sigma_f: float;            (* Signal standard deviation *)
    lengthscales: float array; (* Characteristic lengthscales for each dimension *)
    sigma_n: float;            (* Noise standard deviation *)
  }
  
  (* GP model with training data *)
  type t = {
    params: gp_params;
    x_train: float array array;              (* Training input points *)
    y_train: float array;                    (* Training target values *)
    chol_k: float array array option;        (* Cached Cholesky factor of kernel matrix *)
    alpha: float array option;               (* Cached solution of K^(-1) * (y-m) *)
    log_marginal_likelihood: float option;   (* Cached log marginal likelihood *)
  }
  
  (* Initialize a GP model *)
  let init params =
    {
      params;
      x_train = [||];
      y_train = [||];
      chol_k = None;
      alpha = None;
      log_marginal_likelihood = None;
    }
  
  (* Compute kernel matrix for given inputs *)
  let compute_kernel_matrix gp xs =
    let n = Array.length xs in
    let k = Array.make_matrix n n 0.0 in
    
    for i = 0 to n - 1 do
      for j = i to n - 1 do
        let kij = matern52_kernel xs.(i) xs.(j) gp.params.lengthscales gp.params.sigma_f in
        k.(i).(j) <- kij;
        if i <> j then k.(j).(i) <- kij;
      done;
    done;
    
    k
  
  (* Calculate log marginal likelihood *)
  let calculate_log_marginal_likelihood gp x_train y_train =
    (* Compute kernel matrix K *)
    let k = compute_kernel_matrix gp x_train in
    
    (* Add noise to diagonal: K + sigma_n^2 * I *)
    let k_noisy = Matrix.add_diagonal k (gp.params.sigma_n ** 2.0) in
    
    (* Compute Cholesky decomposition of K *)
    let l = Matrix.cholesky k_noisy in
    
    (* Center the targets around the mean *)
    let y_centered = Array.map (fun y -> y -. gp.params.mean) y_train in
    
    (* Solve (K + sigma_n^2 * I) * alpha = (y - m) *)
    let alpha = Matrix.solve_cholesky l y_centered in
    
    (* Compute log marginal likelihood *)
    let n = Array.length y_train in
    let term1 = -0.5 *. 
      Array.mapi (fun i yi -> yi *. alpha.(i)) y_centered
      |> Array.fold_left (+.) 0.0
    in
    let term2 = -0.5 *. Matrix.log_det_cholesky l in
    let term3 = -0.5 *. float_of_int n *. log (2.0 *. Float.pi) in
    
    (term1 +. term2 +. term3, l, alpha)
  
  (* Fit GP to training data *)
  let fit gp x_train y_train =
    (* Calculate log marginal likelihood and intermediate results *)
    let (log_ml, chol_k, alpha) = calculate_log_marginal_likelihood gp x_train y_train in
    
    { gp with
      x_train;
      y_train;
      chol_k = Some chol_k;
      alpha = Some alpha;
      log_marginal_likelihood = Some log_ml;
    }
  
  (* Predict using fitted GP *)
  let predict gp x =
    match gp.alpha, gp.chol_k with
    | Some alpha, Some chol_k ->
        (* Compute vector k(x, X) *)
        let k_x = Array.map (fun x_i -> 
          matern52_kernel x x_i gp.params.lengthscales gp.params.sigma_f
        ) gp.x_train in
        
        (* Posterior mean: m(x) + k(x,X)^T * alpha *)
        let mean_pred = gp.params.mean +.
          Array.mapi (fun i ki -> ki *. alpha.(i)) k_x
          |> Array.fold_left (+.) 0.0
        in
        
        (* Compute posterior variance if needed *)
        (* This would involve solving for v = L^(-1) * k_x and computing
           var = k(x,x) - v^T * v *)
        
        mean_pred
    | _ -> 
        (* Return prior mean if not fitted *)
        gp.params.mean
  
  (* Optimize hyperparameters by maximizing log marginal likelihood *)
  let optimize_hyperparameters gp x_train y_train =    
    let best_gp = ref gp in
    let best_log_ml = ref (match gp.log_marginal_likelihood with
      | Some ml -> ml
      | None -> Float.neg_infinity
    ) in
    
    (* Grid for sigma_f *)
    let sigma_f_values = [| 0.5; 1.0; 2.0 |] in
    
    (* Grid for lengthscales - scale around initial values *)
    let lengthscale_factors = [| 0.5; 1.0; 2.0 |] in
    
    (* Try different hyperparameter combinations *)
    for sigma_f_idx = 0 to Array.length sigma_f_values - 1 do
      let sigma_f = sigma_f_values.(sigma_f_idx) in
      
      for ls_factor_idx = 0 to Array.length lengthscale_factors - 1 do
        let ls_factor = lengthscale_factors.(ls_factor_idx) in
        
        (* Create new lengthscales *)
        let lengthscales = 
          Array.map (fun ls -> ls *. ls_factor) gp.params.lengthscales 
        in
        
        (* Create new parameters *)
        let params = {
          gp.params with
          sigma_f;
          lengthscales;
        } in
        
        (* Create new GP *)
        let new_gp = { gp with params } in
        
        (* Fit and compute log marginal likelihood *)
        let fitted_gp = fit new_gp x_train y_train in
        
        (* Check if improved *)
        match fitted_gp.log_marginal_likelihood with
        | Some log_ml when log_ml > !best_log_ml ->
            best_gp := fitted_gp;
            best_log_ml := log_ml;
        | _ -> ()
      done
    done;
    
    !best_gp
end