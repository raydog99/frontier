open Torch

type signal = {
  data: Tensor.t;
  mean: float;
  std: float;
}

type returns = {
  data: Tensor.t;
  mean: float;
  std: float;
}

type strategy = {
  signals: signal;
  returns: returns;
  correlation: float;
}

type moment_tensor = {
  data: Tensor.t;
  order: int;
  indices: int array array;
}

type strategy_params = {
  window: int option;
  alpha: float option;
  regularization: float option;
  lookback: int;
}

type cost_model = {
  fixed_cost: float;
  proportional_cost: float;
  quadratic_cost: float;
}

module Isserlis = struct
  (* Generate all possible pairings for 2n elements *)
  let rec generate_pairings n =
    if n = 0 then [[]]
    else if n = 1 then [[(0, 1)]]
    else
      let prev = generate_pairings (n-1) in
      let result = ref [] in
      List.iter (fun pairs ->
        for i = 0 to 2*n-2 do
          let valid = not (List.exists (fun (a,b) -> a = i || b = i) pairs) in
          if valid then
            result := ((2*n-1, i) :: pairs) :: !result
        done
      ) prev;
      !result

  (* Calculate moment using Isserlis theorem *)
  let calculate_moment covariance indices =
    let n = Array.length indices in
    if n mod 2 = 1 then 0.0  (* Odd moments are zero *)
    else
      let pairings = generate_pairings (n/2) in
      List.fold_left (fun acc pairs ->
        let term = List.fold_left (fun prod (i,j) ->
          let idx1 = indices.(i) in
          let idx2 = indices.(j) in
          prod *. covariance.(idx1).(idx2)
        ) 1.0 pairs in
        acc +. term
      ) 0.0 pairings

  (* Calculate arbitrary order moments *)
  let calculate_moments covariance order =
    let dim = Array.length covariance in
    let tensor = Array.make_matrix dim dim 0.0 in
    for i = 0 to dim-1 do
      for j = 0 to dim-1 do
        let indices = Array.init order (fun k -> 
          if k mod 2 = 0 then i else j) in
        tensor.(i).(j) <- calculate_moment covariance indices
      done
    done;
    tensor
end

(* Linear Gaussian *)
module LinearGaussian = struct
  type dimensionless_stats = {
    sharpe: float;
    skewness: float;
    kurtosis: float;
  }

  (* Calculate moments for normal product *)
  let calculate_moments correlation =
    let raw1 = correlation in
    let raw2 = 1.0 +. 2.0 *. correlation ** 2.0 in
    let raw3 = 3.0 *. correlation *. (3.0 +. 2.0 *. correlation ** 2.0) in
    let raw4 = 3.0 *. (3.0 +. 24.0 *. correlation ** 2.0 +. 
                       8.0 *. correlation ** 4.0) in

    (* Convert to standardized moments *)
    let variance = raw2 -. raw1 ** 2.0 in
    let std = sqrt variance in
    let skewness = (raw3 -. 3.0 *. raw1 *. raw2 +. 
                    2.0 *. raw1 ** 3.0) /. (std ** 3.0) in
    let kurtosis = (raw4 -. 4.0 *. raw1 *. raw3 +. 
                    6.0 *. raw1 ** 2.0 *. raw2 -. 
                    3.0 *. raw1 ** 4.0) /. (std ** 4.0) in
    
    (* Calculate dimensionless statistics *)
    let sharpe = correlation /. sqrt(1.0 +. correlation ** 2.0) in
    
    {sharpe; skewness; kurtosis}

  (* Maximal attainable statistics *)
  let maximal_stats = {
    sharpe = sqrt 2.0 /. 2.0;  (* ≈ 0.707 *)
    skewness = 2.0 *. sqrt 2.0;  (* ≈ 2.828 *)
    kurtosis = 15.0;
  }
end

(* Product distribution properties *)
module ProductDistribution = struct
  (* Calculate product density *)
  let density correlation x =
    let k0 = 1.0 /. Float.pi in
    let integrand t =
      exp(-0.5 *. (t *. t +. (x/.t) *. (x/.t) -. 
                   2.0 *. correlation *. t *. (x/.t))) /. 
      abs_float t in
    
    (* Numerical integration *)
    let n = 1000 in
    let dt = 10.0 /. float_of_int n in
    let sum = ref 0.0 in
    
    for i = 0 to n - 1 do
      let t = -5.0 +. float_of_int i *. dt in
      let weight = if i = 0 || i = n then 1.0
                  else if i mod 2 = 0 then 2.0
                  else 4.0 in
      sum := !sum +. weight *. integrand t
    done;
    
    k0 *. !sum *. dt /. 3.0

  (* Non-central chi-square limit *)
  let chi2_limit correlation x =
    let df = 1.0 in
    let lambda = correlation ** 2.0 in
    
    let rec factorial n =
      if n <= 1 then 1.0
      else n *. factorial (n -. 1.0)
    in
    
    let rec sum_terms k acc =
      let term = exp(-lambda/.2.0) *. (lambda/.2.0) ** float_of_int k /.
                factorial k *. x ** ((df +. 2.0 *. float_of_int k -. 2.0)/.2.0) *.
                exp(-x/.2.0) /. 
                (2.0 ** ((df +. 2.0 *. float_of_int k)/.2.0) *. 
                 factorial ((df +. 2.0 *. float_of_int k -. 2.0)/.2.0)) in
      
      if term < 1e-10 || k > 20 then acc
      else sum_terms (k + 1) (acc +. term)
    in
    
    sum_terms 0 0.0
end

(* Optimization and standard errors *)
module TLS = struct
  type tls_result = {
    beta: Tensor.t;
    residuals: Tensor.t;
    correlation: float;
    degrees_of_freedom: float;
  }

  (* Total Least Squares with regularization *)
  let solve_regularized signals returns lambda =
    let n = Tensor.size signals 0 in
    let p = Tensor.size signals 1 in
    
    (* Form augmented matrices *)
    let xy = Tensor.cat [signals; returns] 1 in
    let reg = Tensor.mul_scalar (Tensor.eye p) (sqrt lambda) in
    let aug_matrix = Tensor.cat [xy; reg] 0 in
    
    (* SVD decomposition *)
    let u, s, v = Tensor.svd aug_matrix ~some:false in
    
    (* Extract solution *)
    let v_last = Tensor.select v 1 p in
    let beta = Tensor.narrow v_last 0 0 p in
    let beta_norm = Tensor.get v_last [|p|] in
    let normalized_beta = Tensor.div beta beta_norm in
    
    (* Calculate residuals *)
    let predicted = Tensor.mm signals 
      (Tensor.unsqueeze normalized_beta 1) in
    let residuals = Tensor.sub returns predicted in
    
    (* Calculate correlation *)
    let correlation = Tensor.corrcoef predicted returns |>
                     fun t -> Tensor.get t [|0;1|] in
    
    (* Calculate degrees of freedom *)
    let df = float_of_int n -. (1. +. lambda) *. float_of_int p in
    
    {beta = normalized_beta; residuals; correlation; 
     degrees_of_freedom = df}

  (* Cross validation *)
  let cross_validate signals returns k_folds lambda_range =
    let n = Tensor.size signals 0 in
    let fold_size = n / k_folds in
    let best_lambda = ref (List.hd lambda_range) in
    let min_error = ref infinity in
    
    List.iter (fun lambda ->
      let mut_error = ref 0.0 in
      
      for k = 0 to k_folds - 1 do
        let start_idx = k * fold_size in
        let end_idx = min (start_idx + fold_size) n in
        
        (* Split data *)
        let train_signals = Tensor.cat 
          [Tensor.narrow signals 0 0 start_idx;
           Tensor.narrow signals 0 end_idx (n - end_idx)] 0 in
        let train_returns = Tensor.cat
          [Tensor.narrow returns 0 0 start_idx;
           Tensor.narrow returns 0 end_idx (n - end_idx)] 0 in
        let test_signals = Tensor.narrow signals 0 start_idx fold_size in
        let test_returns = Tensor.narrow returns 0 start_idx fold_size in
        
        (* Fit model *)
        let result = solve_regularized train_signals train_returns lambda in
        
        (* Calculate validation error *)
        let predicted = Tensor.mm test_signals 
          (Tensor.unsqueeze result.beta 1) in
        let error = Tensor.sub predicted test_returns |>
                   Tensor.norm |>
                   Tensor.float_value in
        mut_error := !mut_error +. error
      done;
      
      let avg_error = !mut_error /. float_of_int k_folds in
      if avg_error < !min_error then (
        min_error := avg_error;
        best_lambda := lambda
      )
    ) lambda_range;
    
    !best_lambda
end

(* Standard errors *)
module StandardErrors = struct
  type standard_errors = {
    sharpe_stderr: float;
    skewness_stderr: float;
    kurtosis_stderr: float;
  }

  (* Calculate implied standard errors *)
  let calc_implied_stderrs strategy sample_size =
    let rho = strategy.correlation in
    let t = float_of_int sample_size in
    
    (* Sharpe ratio stderr *)
    let sharpe_stderr = 
      1.0 /. ((rho *. rho +. 1.0) ** 1.5) *.
      sqrt ((1.0 -. rho *. rho) /. (t -. 2.0)) in
    
    (* Skewness stderr *)
    let skewness_stderr =
      -6.0 *. (rho *. rho -. 1.0) /.
      ((rho *. rho +. 1.0) ** 2.5) *.
      sqrt ((1.0 -. rho *. rho) /. (t -. 2.0)) in
    
    (* Kurtosis stderr *)
    let kurtosis_stderr =
      -48.0 *. rho *. (rho *. rho -. 1.0) /.
      ((rho *. rho +. 1.0) ** 3.0) *.
      sqrt ((1.0 -. rho *. rho) /. (t -. 2.0)) in
    
    {sharpe_stderr; skewness_stderr; kurtosis_stderr}

  (* Calculate finite sample corrections *)
  let finite_sample_correction stderr sample_size =
    let t = float_of_int sample_size in
    stderr *. (1.0 +. 5.0 /. (4.0 *. t))

  (* Calculate confidence intervals *)
  let confidence_intervals strategy sample_size confidence =
    let stderrs = calc_implied_stderrs strategy sample_size in
    
    let z = match confidence with
      | 0.90 -> 1.645
      | 0.95 -> 1.96
      | 0.99 -> 2.576 in
    
    let stats = LinearGaussian.calculate_moments strategy.correlation in
    
    let make_interval point stderr =
      (point -. z *. stderr, point +. z *. stderr) in
    
    let sharpe_ci = make_interval stats.sharpe stderrs.sharpe_stderr in
    let skew_ci = make_interval stats.skewness stderrs.skewness_stderr in
    let kurt_ci = make_interval stats.kurtosis stderrs.kurtosis_stderr in
    
    (sharpe_ci, skew_ci, kurt_ci)
end

(* Portfolio analysis *)
module Portfolio = struct
  (* Types for portfolio analysis *)
  type portfolio_stats = {
    sharpe: float;
    skewness: float;
    kurtosis: float;
    correlation_matrix: Tensor.t;
  }

  (* Canonical correlation analysis *)
  module CCA = struct
    type cca_result = {
      correlations: Tensor.t;
      signal_weights: Tensor.t;
      return_weights: Tensor.t;
    }

    (* Solve generalized eigenvalue problem *)
    let solve_gep a b =
      let b_inv_sqrt = 
        let eigenvals, eigenvecs = Tensor.eig b in
        let d_sqrt = Tensor.sqrt eigenvals in
        let d_sqrt_inv = Tensor.div_scalar d_sqrt 1.0 in
        let d_mat = Tensor.diag d_sqrt_inv in
        Tensor.mm (Tensor.mm eigenvecs d_mat) 
          (Tensor.transpose eigenvecs 0 1)
      in
      let m = Tensor.mm (Tensor.mm b_inv_sqrt a) b_inv_sqrt in
      Tensor.eig m

    (* Compute canonical correlations *)
    let compute_correlations signals returns =
      let sxx = Tensor.mm (Tensor.transpose signals 0 1) signals in
      let syy = Tensor.mm (Tensor.transpose returns 0 1) returns in
      let sxy = Tensor.mm (Tensor.transpose signals 0 1) returns in
      
      let m = Tensor.mm 
        (Tensor.mm (Tensor.inverse sxx) sxy)
        (Tensor.mm (Tensor.inverse syy) 
           (Tensor.transpose sxy 0 1)) in
      
      let eigenvals, eigenvecs = Tensor.eig m in
      
      (* Sort eigenvalues and vectors *)
      let n = Tensor.size eigenvals 0 in
      let sorted_indices = Array.init n (fun i -> i) in
      Array.sort (fun i j ->
        compare (Tensor.get eigenvals [|j|])
                (Tensor.get eigenvals [|i|])
      ) sorted_indices;
      
      let sorted_correlations = Tensor.zeros [|n|] in
      let sorted_sig_weights = Tensor.zeros_like eigenvecs in
      
      Array.iteri (fun i idx ->
        Tensor.set sorted_correlations [|i|] 
          (sqrt (Tensor.get eigenvals [|idx|]));
        let vec = Tensor.select eigenvecs 1 idx in
        Tensor.copy_ (Tensor.select sorted_sig_weights 1 i) vec
      ) sorted_indices;
      
      (* Compute return weights *)
      let return_weights = Tensor.mm
        (Tensor.mm (Tensor.inverse syy) 
           (Tensor.transpose sxy 0 1))
        sorted_sig_weights in
      
      {correlations = sorted_correlations;
       signal_weights = sorted_sig_weights;
       return_weights = return_weights}
  end

  (* Multiple asset moment calculations *)
  module MultiAsset = struct
    (* Calculate N-asset MGF *)
    let mgf_n_assets n rho t =
      (1.0 -. 2.0 *. t *. rho -. 2.0 *. t ** 2.0 *. 
       (1.0 -. rho ** 2.0)) ** (-. float_of_int n /. 2.0)

    (* Calculate N-asset moments *)
    let n_asset_moments n rho =
      let n_float = float_of_int n in
      let mu1 = n_float *. rho in
      let mu2 = n_float *. (rho ** 2.0 +. 1.0) in
      let mu3 = n_float *. (n_float +. 2.0) *. rho ** 3.0 *. 
                ((n_float +. 1.0) *. rho ** 2.0 +. 3.0) in
      let mu4 = n_float *. (n_float +. 2.0) *. 
                (n_float +. 4.0) *. (n_float +. 6.0) *. rho ** 4.0 +.
                3.0 *. n_float *. (n_float +. 2.0) *. 
                (1.0 -. rho ** 2.0) ** 2.0 +.
                6.0 *. n_float *. (n_float +. 4.0) *. 
                (n_float +. 2.0) *. rho ** 2.0 *. 
                (1.0 -. rho ** 2.0) in
      [|mu1; mu2; mu3; mu4|]

    (* Calculate maximal N-asset statistics *)
    let n_asset_maximal_stats n =
      let n_float = float_of_int n in
      let max_sharpe = sqrt n_float *. sqrt 2.0 /. 2.0 in
      let max_skewness = 2.0 *. sqrt 2.0 /. sqrt n_float in
      let max_kurtosis = 15.0 in
      {LinearGaussian.sharpe = max_sharpe; 
       skewness = max_skewness; 
       kurtosis = max_kurtosis}
  end

  (* Portfolio optimization *)
  module Optimization = struct
    (* Risk decomposition *)
    let risk_decomposition strategies weights =
      let n = Array.length strategies in
      let contribs = Array.make n 0.0 in
      let total_risk = ref 0.0 in
      
      (* Calculate total portfolio risk *)
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let cov_ij = strategies.(i).correlation *. 
                      strategies.(j).correlation in
          total_risk := !total_risk +. weights.(i) *. weights.(j) *. cov_ij
        done
      done;
      
      (* Calculate marginal contributions *)
      for i = 0 to n - 1 do
        let mrc = ref 0.0 in
        for j = 0 to n - 1 do
          let cov_ij = strategies.(i).correlation *. 
                      strategies.(j).correlation in
          mrc := !mrc +. weights.(j) *. cov_ij
        done;
        contribs.(i) <- weights.(i) *. !mrc /. !total_risk
      done;
      
      contribs

    (* Risk parity optimization *)
    let risk_parity_weights strategies =
      let n = Array.length strategies in
      let target_risk = 1.0 /. float_of_int n in
      let weights = Array.make n (1.0 /. float_of_int n) in
      let max_iter = 100 in
      let tolerance = 1e-6 in
      
      let rec iterate iter prev_weights =
        if iter >= max_iter then prev_weights
        else
          let contribs = risk_decomposition strategies prev_weights in
          
          (* Update weights *)
          let new_weights = Array.mapi (fun i w ->
            w *. sqrt(target_risk /. contribs.(i))
          ) prev_weights in
          
          (* Normalize *)
          let sum = Array.fold_left (+.) 0.0 new_weights in
          let normalized = Array.map (fun w -> w /. sum) new_weights in
          
          (* Check convergence *)
          let max_diff = Array.fold_left2 (fun acc w1 w2 ->
            max acc (abs_float (w1 -. w2))
          ) 0.0 prev_weights normalized in
          
          if max_diff < tolerance then normalized
          else iterate (iter + 1) normalized
      in
      
      iterate 0 weights
  end
end

(* Extensions and applications *)
module Extensions = struct
  (* Transaction costs analysis *)
  module TransactionCosts = struct
    (* Calculate transaction costs *)
    let calculate_costs trades cost_model =
      let n = Array.length trades in
      let total_cost = ref 0.0 in
      
      for i = 0 to n - 1 do
        let trade = abs_float trades.(i) in
        if trade > 0.0 then
          total_cost := !total_cost +. cost_model.fixed_cost +.
                       cost_model.proportional_cost *. trade +.
                       cost_model.quadratic_cost *. trade ** 2.0
      done;
      !total_cost

    (* Optimize with transaction costs *)
    let optimize_with_costs strategy costs initial_weight =
      let rho = strategy.correlation in
      let c = costs.proportional_cost in
      let q = costs.quadratic_cost in
      
      (* Modified Sharpe ratio with costs *)
      let modified_sharpe w =
        let expected_return = rho *. w in
        let variance = w ** 2.0 *. (1.0 +. rho ** 2.0) in
        let expected_cost = c *. abs_float(w -. initial_weight) +.
                          q *. (w -. initial_weight) ** 2.0 in
        (expected_return -. expected_cost) /. sqrt variance
      in
      
      (* Grid search *)
      let n_points = 1000 in
      let best_weight = ref initial_weight in
      let best_sharpe = ref (modified_sharpe initial_weight) in
      
      for i = 0 to n_points do
        let w = -2.0 +. 4.0 *. float_of_int i /. float_of_int n_points in
        let sharpe = modified_sharpe w in
        if sharpe > !best_sharpe then (
          best_sharpe := sharpe;
          best_weight := w
        )
      done;
      
      (!best_weight, !best_sharpe)
  end

  (* Multiple period framework *)
  module MultiPeriod = struct
    (* Calculate autocorrelation function *)
    let autocorrelation strategy lags =
      let rho = strategy.correlation in
      Array.init lags (fun k ->
        let lag = float_of_int (k + 1) in
        rho ** lag
      )

    (* Calculate multi-period moments *)
    let multi_period_moments strategy periods =
      let rho = strategy.correlation in
      let t = float_of_int periods in
      
      (* Scale moments with time *)
      let raw1 = t *. rho in
      let raw2 = t *. (1.0 +. 2.0 *. rho ** 2.0) in
      let raw3 = t *. 3.0 *. rho *. (3.0 +. 2.0 *. rho ** 2.0) in
      let raw4 = t *. 3.0 *. (3.0 +. 24.0 *. rho ** 2.0 +. 
                              8.0 *. rho ** 4.0) in
      
      (* Calculate standardized moments *)
      let variance = raw2 -. raw1 ** 2.0 in
      let std = sqrt variance in
      let skewness = (raw3 -. 3.0 *. raw1 *. raw2 +. 
                     2.0 *. raw1 ** 3.0) /. (std ** 3.0) in
      let kurtosis = (raw4 -. 4.0 *. raw1 *. raw3 +. 
                     6.0 *. raw1 ** 2.0 *. raw2 -. 
                     3.0 *. raw1 ** 4.0) /. (std ** 4.0) in
      
      {mean = raw1; variance; skewness; kurtosis;
       raw_moments = [|raw1; raw2; raw3; raw4|]}

    (* Calculate effective number of observations *)
    let effective_observations strategy sample_size =
      let acf = autocorrelation strategy (sample_size / 10) in
      let sum_acf = 1.0 +. 2.0 *. Array.fold_left (+.) 0.0 acf in
      float_of_int sample_size /. sum_acf
  end

  (* Practical applications *)
  module Applications = struct
    (* Required sample size *)
    let required_sample_size target_sr power alpha =
      let max_sr = LinearGaussian.maximal_stats.sharpe in
      if target_sr > max_sr then None
      else
        let z_alpha = match alpha with
          | 0.01 -> 2.576
          | 0.05 -> 1.96
          | 0.10 -> 1.645 in
        
        let z_beta = match power with
          | 0.80 -> 0.842
          | 0.90 -> 1.282
          | 0.95 -> 1.645 in
        
        let n = ceil ((z_alpha +. z_beta) ** 2.0 *. 
                     (1.0 -. target_sr ** 2.0) /. 
                     target_sr ** 2.0) in
        Some (int_of_float n)

    (* Strategy capacity estimation *)
    let estimate_capacity strategy market_impact =
      let sr = LinearGaussian.calculate_moments(
                 strategy.correlation).sharpe in
      let daily_vol = 0.01 in
      
      (* Maximum AUM before impact degrades SR by 50% *)
      let capacity = sr *. daily_vol /. 
                    (2.0 *. sqrt (252.0) *. market_impact) in
      capacity *. 1e6
  end
end

  (* Strategy creation and analysis *)
  module Creation = struct
    (* Create signal from parameters *)
    let create_signal returns params =
      match params with
      | {window = Some w; alpha = None; _} ->
          let n = Tensor.size returns 0 in
          let signal = Tensor.zeros [|n|] in
          
          for i = w to n - 1 do
            let slice = Tensor.narrow returns 0 (i - w) w in
            let ma = Tensor.mean slice in
            Tensor.set signal [|i|] ma
          done;
          signal

      | {window = None; alpha = Some a; _} ->
          let n = Tensor.size returns 0 in
          let signal = Tensor.zeros [|n|] in
          
          Tensor.set signal [|0|] 
            (Tensor.get returns [|0|]);
          
          for i = 1 to n - 1 do
            let prev = Tensor.get signal [|i-1|] in
            let curr = Tensor.get returns [|i|] in
            let new_val = a *. curr +. (1. -. a) *. prev in
            Tensor.set signal [|i|] new_val
          done;
          signal

      | _ -> failwith "Invalid signal parameters"

    (* Create complete strategy *)
    let create_strategy returns params =
      (* Generate base signal *)
      let signal = create_signal returns params in
      
      (* Optimize if regularization specified *)
      let final_signal = match params.regularization with
        | Some lambda ->
            let tls = TLS.solve_regularized signal returns lambda in
            tls.beta
        | None -> signal in
      
      (* Calculate correlation *)
      let correlation = Tensor.corrcoef final_signal returns |>
                       fun t -> Tensor.get t [|0;1|] in
      
      {
        signals = {
          data = final_signal;
          mean = Tensor.mean final_signal |> 
                 Tensor.float_value;
          std = Tensor.std final_signal |> 
                Tensor.float_value;
        };
        returns = {
          data = returns;
          mean = Tensor.mean returns |> 
                 Tensor.float_value;
          std = Tensor.std returns |> 
                Tensor.float_value;
        };
        correlation;
      }
  end

(* Strategy analysis *)
module Analysis = struct
  type analysis_result = {
    moments: LinearGaussian.dimensionless_stats;
    standard_errors: standard_errors;
    relative_performance: float * float * float;  (* vs maximal *)
    confidence_intervals: (float * float) * (float * float) * (float * float);
  }

  (* Analyze strategy performance *)
  let analyze_strategy strategy sample_size confidence =
    (* Calculate moments *)
    let moments = LinearGaussian.calculate_moments strategy.correlation in
    
    (* Calculate standard errors *)
    let stderrs = calc_implied_stderrs strategy sample_size in
    
    (* Calculate relative performance *)
    let max_stats = LinearGaussian.maximal_stats in
    let rel_perf = (
      moments.sharpe /. max_stats.sharpe,
      moments.skewness /. max_stats.skewness,
      moments.kurtosis /. max_stats.kurtosis
    ) in
    
    (* Calculate confidence intervals *)
    let cis = confidence_intervals strategy sample_size confidence in
    
    {
      moments;
      standard_errors = stderrs;
      relative_performance = rel_perf;
      confidence_intervals = cis;
    }

  (* Calculate risk decomposition *)
  let risk_decomposition strategies weights =
    Portfolio.Optimization.risk_decomposition strategies weights
end

(* Portfolio construction *)
module PortfolioConstruction = struct
  type portfolio_result = {
    weights: float array;
    statistics: Portfolio.portfolio_stats;
    risk_contributions: float array;
  }

  (* Create optimal portfolio *)
  let create_optimal_portfolio strategies =
    (* Calculate optimal weights *)
    let weights = Portfolio.Optimization.risk_parity_weights strategies in
    
    (* Calculate portfolio statistics *)
    let n = Array.length strategies in
    let correlations = Array.make_matrix n n 0.0 in
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        correlations.(i).(j) <- strategies.(i).correlation *. 
                               strategies.(j).correlation
      done
    done;
    
    let corr_tensor = Tensor.of_float_array2 correlations 
                       [|n; n|] in
    
    (* Calculate portfolio moments *)
    let port_moments = MultiAsset.n_asset_moments n 
      (Array.fold_left2 (fun acc w s -> 
        acc +. w *. s.correlation) 0.0 weights strategies) in
    
    let statistics = {
      sharpe = port_moments.(0) /. sqrt port_moments.(1);
      skewness = port_moments.(2) /. (port_moments.(1) ** 1.5);
      kurtosis = port_moments.(3) /. (port_moments.(1) ** 2.0);
      correlation_matrix = corr_tensor;
    } in
    
    (* Calculate risk contributions *)
    let risk_contribs = Analysis.risk_decomposition strategies weights in
    
    {weights; statistics; risk_contributions = risk_contribs}
end

(* Transaction cost analysis *)
module CostAnalysis = struct
  (* Analyze strategy with costs *)
  let analyze_with_costs strategy trades cost_model =
    (* Calculate base costs *)
    let costs = TransactionCosts.calculate_costs trades cost_model in
    
    (* Calculate modified Sharpe ratio *)
    let (opt_weight, opt_sharpe) = 
      TransactionCosts.optimize_with_costs strategy cost_model 
        (Array.get trades 0) in
    
    (* Calculate optimal rebalancing period *)
    let rebal_period = 
      Extensions.Applications.optimal_rebalancing_period strategy 
        cost_model in
    
    (costs, opt_weight, opt_sharpe, rebal_period)

  (* Estimate capacity *)
  let estimate_capacity strategy market_impact =
    Extensions.Applications.estimate_capacity strategy market_impact
end