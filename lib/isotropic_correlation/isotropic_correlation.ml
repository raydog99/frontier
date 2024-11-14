open Torch

module MatrixOps = struct
  let compute_grand_sum tensor =
    Tensor.sum tensor ~dim:[0; 1] ~keepdim:false
    |> Tensor.to_float0_exn

  let compute_trace tensor =
    Tensor.diagonal tensor 0
    |> Tensor.sum ~dim:[0] ~keepdim:false
    |> Tensor.to_float0_exn

  let tensor_of_ones n = Tensor.ones [n; 1]

  let diagonal_matrix values =
    let n = Tensor.shape values |> List.hd in
    let diag = Tensor.zeros [n; n] in
    Tensor.diagonal_scatter_ diag values 0;
    diag

  let verify_matrix_properties matrix =
    let n = Tensor.size matrix 0 in
    
    (* Check symmetry *)
    let is_symmetric = 
      let diff = Tensor.sub matrix (Tensor.transpose2 matrix) in
      let max_diff = 
        Tensor.abs diff |> Tensor.max |> Tensor.to_float0_exn in
      max_diff < 1e-10 in
    
    (* Check positive definiteness *)
    let is_positive_definite =
      try 
        let _ = Tensor.linalg_cholesky matrix in 
        true
      with _ -> false in
    
    (* Check unit diagonal *)
    let diag = Tensor.diagonal matrix 0 in
    let has_unit_diagonal = 
      let diag_diff = Tensor.sub diag (Tensor.ones [n]) in
      let max_diag_diff = 
        Tensor.abs diag_diff |> Tensor.max |> Tensor.to_float0_exn in
      max_diag_diff < 1e-10 in
    
    (* Check correlation bounds *)
    let valid_correlations =
      let min_corr = Tensor.min matrix |> Tensor.to_float0_exn in
      let max_corr = Tensor.max matrix |> Tensor.to_float0_exn in
      min_corr >= -1. && max_corr <= 1. in
    
    {is_symmetric; is_positive_definite; has_unit_diagonal; valid_correlations}
end

module IsotropicCoreImpl : IsotropicCore = struct
  let create_correlation_matrix rho n =
    if n <= 0 then 
      invalid_arg "Matrix size must be positive";
    if rho < -1. /. float_of_int (max 1 (n-1)) || rho > 1. then
      invalid_arg "Invalid correlation coefficient";
    
    let ones = Tensor.ones [n; n] in
    let eye = Tensor.eye n in
    Tensor.add eye (Tensor.mul_scalar (Tensor.sub ones eye) rho)

  let create_covariance_matrix sigma rho n =
    if sigma <= 0. then
      invalid_arg "Sigma must be positive";
    let corr = create_correlation_matrix rho n in
    Tensor.mul_scalar corr (sigma *. sigma)

  let compute_portfolio_variance ~weights ~cov =
    let result = 
      Tensor.mm (Tensor.mm (Tensor.transpose2 weights) cov) weights in
    Tensor.to_float0_exn result

  let homoskedastic_portfolio_variance ~sigma ~rho ~weights =
    let n = Tensor.size weights 0 in
    let n_float = float_of_int n in
    
    let sum_w = 
      Tensor.sum weights ~dim:[0] ~keepdim:false 
      |> Tensor.to_float0_exn in
    let sum_w2 = 
      Tensor.pow weights 2.
      |> Tensor.sum ~dim:[0] ~keepdim:false
      |> Tensor.to_float0_exn in
    
    let total_variance = 
      sigma *. sigma *. (
        sum_w2 *. (1. -. rho) +.
        sum_w *. sum_w *. rho
      ) in
    
    let systematic_component = 
      sigma *. sigma *. sum_w *. sum_w *. rho in
    
    let residual_component = 
      sigma *. sigma *. sum_w2 *. (1. -. rho) in
    
    let effective_dof = 
      n_float /. (1. +. (n_float -. 1.) *. rho) in
    
    {total_variance; systematic_component; 
     residual_component; effective_dof}

  let compute_factor_decomposition ~returns =
    let n = Tensor.size returns 1 in
    let n_float = float_of_int n in
    
    (* Compute market factor *)
    let market_weights = 
      Tensor.div_scalar (Tensor.ones [1; n]) n_float in
    let market_factor = Tensor.mm market_weights returns in
    
    (* Compute factor loadings *)
    let market_var = 
      Tensor.var market_factor ~dim:[1] ~unbiased:true ~keepdim:false in
    let covariance = 
      Tensor.mm returns (Tensor.transpose2 market_factor) in
    let factor_loadings = Tensor.div covariance market_var in
    
    (* Compute residual returns *)
    let market_component = 
      Tensor.mm (Tensor.transpose2 factor_loadings) market_factor in
    let residual_returns = Tensor.sub returns market_component in
    
    (* Compute risk components *)
    let systematic_risk = 
      Tensor.var market_component ~dim:[1] ~unbiased:true ~keepdim:false
      |> Tensor.to_float0_exn in
    let residual_risk = 
      Tensor.var residual_returns ~dim:[1] ~unbiased:true ~keepdim:false
      |> Tensor.to_float0_exn in
    
    (* Compute variance ratio *)
    let variance_ratio = residual_risk /. systematic_risk in
    
    {market_factor; factor_loadings; residual_returns;
     systematic_risk; residual_risk; variance_ratio}

  let compute_limiting_behavior ~sigma ~rho ~n =
    let n_float = float_of_int n in
    
    (* Asymptotic portfolio variance *)
    let asymptotic_variance = sigma *. sigma *. rho in
    
    (* Convergence rate to asymptotic variance *)
    let convergence_rate = 1. /. sqrt n_float in
    
    (* Minimum feasible correlation *)
    let min_correlation = -1. /. (n_float -. 1.) in
    
    (* Maximum effective degrees of freedom *)
    let max_effective_dof = 
      if rho <= 0. then infinity
      else 1. /. rho in
    
    {asymptotic_variance; convergence_rate; 
     min_correlation; max_effective_dof}
end

open Torch

module EmpiricalAnalysis = struct
  let analyze_normality ~returns =
    let port_returns = 
      let n = Tensor.size returns 1 in
      let weights = 
        Tensor.div_scalar (Tensor.ones [1; n]) (float_of_int n) in
      Tensor.mm weights returns in
    
    let sorted_returns, _ = 
      Tensor.sort port_returns ~descending:false in
    let n = Tensor.size sorted_returns 1 in
    let n_float = float_of_int n in
    
    (* Shapiro-Wilk test *)
    let mean = 
      Tensor.mean port_returns ~dim:[1] ~keepdim:false
      |> Tensor.to_float0_exn in
    let std = 
      Tensor.std port_returns ~dim:[1] ~unbiased:true ~keepdim:false
      |> Tensor.to_float0_exn in
    
    let normalized = 
      Tensor.div (Tensor.sub sorted_returns mean) std in
    
    let shapiro_wilk =
      let weights = ref [] in
      for i = 0 to n-1 do
        let m = float_of_int i -. n_float *. 0.5 +. 1. in
        let w = exp (-0.5 *. m *. m /. (n_float *. n_float)) in
        weights := w :: !weights
      done;
      let w = Array.of_list !weights in
      let numerator = 
        Array.mapi (fun i x -> 
          x *. Tensor.get normalized [0; i]
        ) w
        |> Array.fold_left (+.) 0. in
      let denominator = 
        Tensor.pow normalized 2.
        |> Tensor.sum ~dim:[1] ~keepdim:false
        |> Tensor.to_float0_exn in
      numerator *. numerator /. denominator in
    
    (* Anderson-Darling test *)
    let anderson_darling =
      let empirical_cdf = 
        Array.init n (fun i -> 
          float_of_int (i + 1) /. n_float) in
      let theoretical_cdf = 
        Array.init n (fun i ->
          0.5 *. (1. +. erf (
            Tensor.get normalized [0; i] /. sqrt 2.
          ))) in
      Array.fold_left2 (fun acc p_i f_i ->
        acc +. (2. *. float_of_int (n-1) +. 1.) *.
        (log p_i +. log (1. -. f_i))
      ) 0. empirical_cdf theoretical_cdf in
    
    (* Kolmogorov-Smirnov test *)
    let max_diff = ref 0. in
    let p_value = ref 1. in
    for i = 0 to n-1 do
      let e_cdf = float_of_int (i + 1) /. n_float in
      let t_cdf = 0.5 *. (1. +. erf (
        Tensor.get normalized [0; i] /. sqrt 2.
      )) in
      max_diff := max !max_diff (abs_float (e_cdf -. t_cdf))
    done;
    let lambda = (!max_diff *. sqrt n_float +. 0.12 +. 
                 0.11 /. sqrt n_float) in
    p_value := exp (-2. *. lambda *. lambda);
    
    (* Q-Q plot data *)
    let qq_plot_data =
      Array.init n (fun i ->
        let theoretical = 
          sqrt 2. *. erfinv (2. *. float_of_int (i + 1) /. n_float -. 1.) in
        let empirical = Tensor.get normalized [0; i] in
        (theoretical, empirical)
      ) in
    
    {
      shapiro_wilk;
      anderson_darling;
      ks_test = (!max_diff, !p_value);
      qq_plot_data;
      is_normal = shapiro_wilk > 0.95 && anderson_darling < 2.492;
      confidence_level = 0.95;
    }

  let analyze_correlation_distribution ~returns =
    let corr = 
      let centered = 
        Tensor.sub returns (Tensor.mean returns ~dim:[0] ~keepdim:true) in
      let std = 
        Tensor.std returns ~dim:[0] ~unbiased:true ~keepdim:true in
      let normalized = Tensor.div centered std in
      Tensor.mm (Tensor.transpose2 normalized) normalized in
    
    let n = Tensor.size returns 1 in
    
    (* Extract correlations *)
    let correlations = ref [] in
    for i = 0 to n-2 do
      for j = i+1 to n-1 do
        correlations := Tensor.get corr [i; j] :: !correlations
      done
    done;
    
    (* Fisher transformation *)
    let z_scores = 
      List.map (fun rho -> 
        0.5 *. log ((1. +. rho) /. (1. -. rho))
      ) !correlations in
    
    let mean_z = 
      List.fold_left (+.) 0. z_scores /. 
      float_of_int (List.length z_scores) in
    let std_z = 
      sqrt (List.fold_left (fun acc z -> 
        acc +. (z -. mean_z) *. (z -. mean_z)
      ) 0. z_scores /. float_of_int (List.length z_scores)) in
    
    (* Rank analysis *)
    let eigvals, _ = Tensor.linalg_eigh corr ~UPLO:'L' in
    let sorted_vals, _ = Tensor.sort eigvals ~descending:true in
    
    let sum_eig = 
      Tensor.sum sorted_vals ~dim:[0] ~keepdim:false
      |> Tensor.to_float0_exn in
    let sum_eig_squared = 
      Tensor.pow sorted_vals 2.
      |> Tensor.sum ~dim:[0] ~keepdim:false
      |> Tensor.to_float0_exn in
    
    let effective_rank = sum_eig *. sum_eig /. sum_eig_squared in
    let participation_ratio = sum_eig_squared /. (sum_eig *. sum_eig) in
    let top_eigenvalue_ratio = Tensor.get sorted_vals [0] /. sum_eig in
    
    let eigenvalue_spacing = 
      Array.init (n-1) (fun i ->
        Tensor.get sorted_vals [i] -. Tensor.get sorted_vals [i+1]
      ) in
    
    (* Isotropy tests *)
    let correlation_variance = 
      List.fold_left (fun acc rho -> acc +. rho *. rho)
        0. !correlations /. float_of_int (List.length !correlations) in
    
    let homogeneity_stat = 
      let sorted_corrs = List.sort compare !correlations in
      let mid = List.length sorted_corrs / 2 in
      let (low, high) = List.partition (fun (_, i) -> i < mid)
        (List.mapi (fun i x -> (x, i)) sorted_corrs) in
      let var_ratio = 
        (List.fold_left (fun acc (x, _) -> acc +. x *. x) 0. high) /.
        (List.fold_left (fun acc (x, _) -> acc +. x *. x) 0. low) in
      abs_float (log var_ratio) in
    
    {
      fisher_stats = {
        mean_z;
        std_z;
        confidence_interval = (
          tanh (mean_z -. 1.96 *. std_z),
          tanh (mean_z +. 1.96 *. std_z)
        );
        normality_test = 
          let n = float_of_int (List.length z_scores) in
          let skewness = List.fold_left (fun acc z ->
            let z_std = (z -. mean_z) /. std_z in
            acc +. z_std *. z_std *. z_std
          ) 0. z_scores /. n in
          let kurtosis = List.fold_left (fun acc z ->
            let z_std = (z -. mean_z) /. std_z in
            acc +. z_std *. z_std *. z_std *. z_std
          ) 0. z_scores /. n in
          n *. (skewness *. skewness /. 6. +. 
                (kurtosis -. 3.) *. (kurtosis -. 3.) /. 24.);
      };
      rank_analysis = {
        effective_rank;
        participation_ratio;
        top_eigenvalue_ratio;
        eigenvalue_spacing;
      };
      isotropy_tests = {
        correlation_variance;
        homogeneity_stat;
        is_isotropic = correlation_variance < 0.1 && homogeneity_stat < 0.1;
      };
    }

  let compute_effective_dof_curve ~returns ~max_size ~samples =
    let curve = ref [] in
    for size = 2 to max_size do
      let dofs = ref [] in
      for _ = 1 to samples do
        let indices = 
          Tensor.randperm (Tensor.size returns 1)
          |> Tensor.narrow 0 0 size in
        let subset = Tensor.index_select returns 1 indices in
        
        let cov = 
          let centered = 
            Tensor.sub subset 
              (Tensor.mean subset ~dim:[0] ~keepdim:true) in
          Tensor.mm (Tensor.transpose2 centered) centered in
        
        let vi = 
          Tensor.diagonal cov 0
          |> Tensor.sum
          |> Tensor.to_float0_exn
          |> fun x -> x /. float_of_int (size * size) in
        
        let weights = 
          Tensor.div_scalar (Tensor.ones [1; size]) 
            (float_of_int size) in
        let vp = 
          Tensor.mm (Tensor.mm weights cov) 
            (Tensor.transpose2 weights)
          |> Tensor.to_float0_exn in
        
        dofs := float_of_int size *. vi /. vp :: !dofs
      done;
      
      let mean_dof = 
        List.fold_left (+.) 0. !dofs /. float_of_int samples in
      curve := (size, mean_dof) :: !curve
    done;
    List.rev !curve

  let validate_isotropy ~returns =
    let n = Tensor.size returns 1 in
    let corr = 
      let centered = 
        Tensor.sub returns (Tensor.mean returns ~dim:[0] ~keepdim:true) in
      let std = 
        Tensor.std returns ~dim:[0] ~unbiased:true ~keepdim:true in
      let normalized = Tensor.div centered std in
      Tensor.mm (Tensor.transpose2 normalized) normalized in
    
    (* Correlation statistics *)
    let correlations = ref [] in
    for i = 0 to n-2 do
      for j = i+1 to n-1 do
        correlations := Tensor.get corr [i; j] :: !correlations
      done
    done;
    
    let mean = 
      List.fold_left (+.) 0. !correlations /. 
      float_of_int (List.length !correlations) in
    let std = 
      sqrt (List.fold_left (fun acc x -> 
        acc +. (x -. mean) *. (x -. mean)
      ) 0. !correlations /. float_of_int (List.length !correlations)) in
    
    let skewness = 
      List.fold_left (fun acc x ->
        let z = (x -. mean) /. std in
        acc +. z *. z *. z
      ) 0. !correlations /. float_of_int (List.length !correlations) in
    
    let kurtosis = 
      List.fold_left (fun acc x ->
        let z = (x -. mean) /. std in
        acc +. z *. z *. z *. z
      ) 0. !correlations /. float_of_int (List.length !correlations) in
    
    (* Eigenvalue analysis *)
    let eigvals, _ = Tensor.linalg_eigh corr ~UPLO:'L' in
    let sorted_vals, _ = Tensor.sort eigvals ~descending:true in
    
    let n_float = float_of_int n in
    let q = n_float /. float_of_int (Tensor.size returns 0) in
    
    (* Marchenko-Pastur edge *)
    let lambda_plus = (1. +. sqrt q) ** 2. in
    let bulk_edge = Tensor.get sorted_vals [1] in
    let spectral_gap = Tensor.get sorted_vals [0] -. bulk_edge in
    let mp_deviation = abs_float (bulk_edge -. lambda_plus) /. lambda_plus in
    
    (* Stability metrics *)
    let condition_number = 
      Tensor.get sorted_vals [0] /. 
      Tensor.get sorted_vals [n-1] in
    
    let effective_rank = 
      let sum_vals = 
        Tensor.sum sorted_vals ~dim:[0] ~keepdim:false
        |> Tensor.to_float0_exn in
      let sum_squared = 
        Tensor.pow sorted_vals 2.
        |> Tensor.sum ~dim:[0] ~keepdim:false
        |> Tensor.to_float0_exn in
      sum_vals *. sum_vals /. sum_squared in
    
    {
      correlation_stats = {mean; std; skewness; kurtosis};
      eigenvalue_stats = {bulk_edge; spectral_gap; mp_deviation};
      stability_metrics = {
        condition_number;
        effective_rank;
        numerical_stability = condition_number < 100.;
      };
    }
end

module CrossValidation = struct
  let validate_isotropic_model ~returns ~folds =
    let n = Tensor.size returns 1 in
    let t = Tensor.size returns 0 in
    let fold_size = t / folds in
    
    let correlation_errors = ref [] in
    let rank_metrics = ref [] in
    let rho_estimates = ref [] in
    
    for fold = 0 to folds-1 do
      (* Split data *)
      let val_start = fold * fold_size in
      let val_end = val_start + fold_size in
      
      let train_returns = 
        Tensor.cat [
          Tensor.narrow returns 0 0 val_start;
          Tensor.narrow returns 0 val_end (t - val_end)
        ] ~dim:0 in
      let val_returns = 
        Tensor.narrow returns 0 val_start fold_size in
      
      (* Compute correlation matrices *)
      let compute_corr returns =
        let centered = 
          Tensor.sub returns (Tensor.mean returns ~dim:[0] ~keepdim:true) in
        let std = 
          Tensor.std returns ~dim:[0] ~unbiased:true ~keepdim:true in
        let normalized = Tensor.div centered std in
        Tensor.mm (Tensor.transpose2 normalized) normalized in
      
      let train_corr = compute_corr train_returns in
      let val_corr = compute_corr val_returns in
      
      (* Estimate rho from training data *)
      let train_correlations = ref [] in
      for i = 0 to n-2 do
        for j = i+1 to n-1 do
          train_correlations := Tensor.get train_corr [i; j] :: !train_correlations
        done
      done;
      
      let rho = 
        List.fold_left (+.) 0. !train_correlations /. 
        float_of_int (List.length !train_correlations) in
      rho_estimates := rho :: !rho_estimates;
      
      (* Create isotropic model *)
      let model_corr = 
        let ones = Tensor.ones [n; n] in
        let eye = Tensor.eye n in
        Tensor.add eye (Tensor.mul_scalar (Tensor.sub ones eye) rho) in
      
      (* Compute validation error *)
      let error = 
        Tensor.sub val_corr model_corr
        |> Tensor.pow (Tensor.float_vec [2.])
        |> Tensor.mean
        |> Tensor.to_float0_exn in
      correlation_errors := error :: !correlation_errors;
      
      (* Compute rank metrics *)
      let eigvals, _ = Tensor.linalg_eigh val_corr ~UPLO:'L' in
      let sorted_vals, _ = Tensor.sort eigvals ~descending:true in
      
      let sum_eig = 
        Tensor.sum sorted_vals ~dim:[0] ~keepdim:false
        |> Tensor.to_float0_exn in
      let sum_eig_squared = 
        Tensor.pow sorted_vals 2.
        |> Tensor.sum ~dim:[0] ~keepdim:false
        |> Tensor.to_float0_exn in
      
      let effective_rank = sum_eig *. sum_eig /. sum_eig_squared in
      rank_metrics := effective_rank :: !rank_metrics
    done;
    
    (* Compute summary statistics *)
    let mean_error = 
      List.fold_left (+.) 0. !correlation_errors /. float_of_int folds in
    let std_error = 
      sqrt (List.fold_left (fun acc x -> 
        acc +. (x -. mean_error) *. (x -. mean_error)
      ) 0. !correlation_errors /. float_of_int (folds - 1)) in
    
    let mean_rank = 
      List.fold_left (+.) 0. !rank_metrics /. float_of_int folds in
    let rank_stability = 
      1. -. (List.fold_left (fun acc x -> 
        acc +. abs_float (x -. mean_rank)
      ) 0. !rank_metrics /. float_of_int folds) /. mean_rank in
    
    let best_rho = 
      List.fold_left (fun acc x -> if x < acc then x else acc) 
        infinity !rho_estimates in
    
    let model_consistency = 
      1. -. std_error /. mean_error in
    
    {
      correlation_scores = {
        mean_error;
        std_error;
        confidence_bounds = (
          mean_error -. 1.96 *. std_error,
          mean_error +. 1.96 *. std_error
        );
      };
      rank_scores = {
        mean_effective_rank = mean_rank;
        rank_stability;
        model_consistency;
      };
      model_selection = {
        best_rho;
        model_evidence = 1. /. mean_error;
        cross_validation_error = mean_error;
      };
    }
end

open Torch

module PortfolioOptimization = struct
  let compute_optimal_weights ~alpha ~sigma ~rho ~lambda ~n =
    (* Compute inverse correlation matrix *)
    let ginv = 
      let factor = 1. /. ((1. -. rho) *. (1. +. (float_of_int (n-1)) *. rho)) in
      let eye = Tensor.eye n in
      let ones = Tensor.ones [n; n] in
      let term1 = Tensor.div_scalar eye (1. -. rho) in
      let term2 = Tensor.mul_scalar ones (rho *. factor) in
      Tensor.sub term1 term2 in
    
    (* Scale by variance *)
    let scaled_ginv = Tensor.div_scalar ginv (sigma *. sigma) in
    
    (* Compute optimal weights *)
    let weights = Tensor.mm scaled_ginv alpha in
    let scaled_weights = Tensor.div_scalar weights (2. *. lambda) in
    
    (* Compute expected return *)
    let exp_return = 
      Tensor.mm (Tensor.transpose2 scaled_weights) alpha
      |> Tensor.to_float0_exn in
    
    (scaled_weights, exp_return)

  let optimize_with_constraints ~alpha ~sigma ~rho ~constraints ~n =
    let initial_weights = 
      let (w, _) = compute_optimal_weights ~alpha ~sigma ~rho ~lambda:0.5 ~n in
      w in
    
    let apply_constraint weights = function
      | `LongOnly ->
          let pos_weights = 
            Tensor.maximum weights (Tensor.zeros_like weights) in
          Tensor.div pos_weights 
            (Tensor.sum pos_weights ~dim:[0] ~keepdim:false)
      
      | `Sector (indices, min_weight, max_weight) ->
          let adjusted = Tensor.clone weights in
          let sector_weight = 
            Array.fold_left (fun acc idx ->
              acc +. Tensor.get adjusted [idx; 0]
            ) 0. indices in
          
          if sector_weight < min_weight || sector_weight > max_weight then
            let target_weight = 
              if sector_weight < min_weight then min_weight
              else max_weight in
            let scale = target_weight /. sector_weight in
            Array.iter (fun idx ->
              let w = Tensor.get adjusted [idx; 0] in
              Tensor.set_ adjusted [idx; 0] (w *. scale)
            ) indices;
          adjusted
      
      | `Tracking (benchmark, max_te) ->
          let diff = Tensor.sub weights benchmark in
          let cov = 
            let corr = 
              let ones = Tensor.ones [n; n] in
              let eye = Tensor.eye n in
              Tensor.add eye 
                (Tensor.mul_scalar (Tensor.sub ones eye) rho) in
            Tensor.mul_scalar corr (sigma *. sigma) in
          
          let te = 
            Tensor.sqrt (
              Tensor.mm (Tensor.mm (Tensor.transpose2 diff) cov) diff
            ) |> Tensor.to_float0_exn in
          
          if te <= max_te then weights
          else
            let scale = max_te /. te in
            Tensor.add benchmark 
              (Tensor.mul_scalar (Tensor.sub weights benchmark) scale)
      
      | `Turnover (current, max_turnover) ->
          let diff = Tensor.sub weights current in
          let turnover = 
            Tensor.sum (Tensor.abs diff) ~dim:[0] ~keepdim:false
            |> Tensor.to_float0_exn in
          
          if turnover <= max_turnover then weights
          else
            let scale = max_turnover /. turnover in
            Tensor.add current (Tensor.mul_scalar diff scale) in
    
    let rec optimize weights iter max_iter tol =
      if iter >= max_iter then weights
      else
        let new_weights = 
          List.fold_left apply_constraint weights constraints in
        
        let diff = 
          Tensor.sub new_weights weights
          |> Tensor.abs
          |> Tensor.max
          |> Tensor.to_float0_exn in
        
        if diff < tol then new_weights
        else optimize new_weights (iter + 1) max_iter tol in
    
    let final_weights = optimize initial_weights 0 1000 1e-6 in
    
    (* Compute objective value *)
    let cov = 
      let corr = 
        let ones = Tensor.ones [n; n] in
        let eye = Tensor.eye n in
        Tensor.add eye (Tensor.mul_scalar (Tensor.sub ones eye) rho) in
      Tensor.mul_scalar corr (sigma *. sigma) in
    
    let obj_value = 
      let expected_return = 
        Tensor.mm (Tensor.transpose2 final_weights) alpha
        |> Tensor.to_float0_exn in
      let risk = 
        Tensor.mm (Tensor.mm (Tensor.transpose2 final_weights) cov) final_weights
        |> Tensor.to_float0_exn in
      expected_return -. 0.5 *. risk in
    
    (final_weights, obj_value)

  let analyze_weight_transition ~alpha ~sigma ~rho ~n_range =
    let compute_weights n =
      let (weights, _) = 
        compute_optimal_weights ~alpha ~sigma ~rho ~lambda:0.5 ~n in
      weights in
    
    let weights_by_n = 
      List.map (fun n -> (n, compute_weights n)) n_range in
    
    (* Find transition point *)
    let transition_point =
      let rec find_transition prev_weights = function
        | [] -> List.hd n_range
        | (n, weights) :: rest ->
            let diff = 
              Tensor.sub weights prev_weights
              |> Tensor.abs
              |> Tensor.mean
              |> Tensor.to_float0_exn in
            if diff < 0.1 then n
            else find_transition weights rest in
      match weights_by_n with
      | [] -> List.hd n_range
      | (_, first_weights) :: rest -> 
          find_transition first_weights rest in
    
    (* Compute convergence rate *)
    let convergence_rate =
      let diffs = 
        List.mapi (fun i (_, w) ->
          if i = 0 then 0.
          else
            let (_, prev_w) = List.nth weights_by_n (i-1) in
            Tensor.sub w prev_w
            |> Tensor.abs
            |> Tensor.mean
            |> Tensor.to_float0_exn
        ) weights_by_n in
      
      1. /. (List.fold_left (+.) 0. diffs /. 
             float_of_int (List.length diffs - 1)) in
    
    let limiting_weights = 
      snd (List.hd (List.rev weights_by_n)) in
    
    {
      weights = weights_by_n;
      transition_point;
      convergence_rate;
      limiting_weights;
    }

  let compute_risk_contributions ~weights ~sigma ~rho ~n =
    let cov = 
      let corr = 
        let ones = Tensor.ones [n; n] in
        let eye = Tensor.eye n in
        Tensor.add eye (Tensor.mul_scalar (Tensor.sub ones eye) rho) in
      Tensor.mul_scalar corr (sigma *. sigma) in
    
    let total_risk = 
      Tensor.sqrt (
        Tensor.mm (Tensor.mm (Tensor.transpose2 weights) cov) weights
      ) |> Tensor.to_float0_exn in
    
    let marginal_risks = 
      Tensor.mm cov weights
      |> Tensor.div_scalar total_risk in
    
    Array.init n (fun i ->
      let w = Tensor.get weights [i; 0] in
      let mr = Tensor.get marginal_risks [i; 0] in
      w *. mr
    )

  let optimize_hierarchy ~alpha ~sigma ~rho ~hierarchy =
    let n = Tensor.size alpha 0 in
    
    (* First optimize within groups *)
    let group_weights = 
      List.map (fun indices ->
        let group_size = List.length indices in
        let group_alpha = 
          Tensor.zeros [group_size; 1] in
        List.iteri (fun i idx ->
          Tensor.set_ group_alpha [i; 0] (Tensor.get alpha [idx; 0])
        ) indices;
        
        let (weights, _) = 
          compute_optimal_weights 
            ~alpha:group_alpha ~sigma ~rho ~lambda:0.5 ~n:group_size in
        (indices, weights)
      ) hierarchy in
    
    (* Then optimize between groups *)
    let n_groups = List.length hierarchy in
    let group_alloc = 
      let group_returns = 
        List.map (fun (indices, weights) ->
          let group_size = List.length indices in
          let sum_weight = 
            Tensor.sum weights ~dim:[0] ~keepdim:false
            |> Tensor.to_float0_exn in
          sum_weight /. float_of_int group_size
        ) group_weights in
      
      let group_alpha = 
        Tensor.float_vec group_returns
        |> Tensor.reshape [n_groups; 1] in
      
      let (alloc, _) = 
        compute_optimal_weights 
          ~alpha:group_alpha ~sigma ~rho ~lambda:0.5 ~n:n_groups in
      alloc in
    
    (* Combine hierarchical weights *)
    let final_weights = Tensor.zeros [n; 1] in
    List.iteri (fun i (indices, weights) ->
      let group_weight = Tensor.get group_alloc [i; 0] in
      List.iteri (fun j idx ->
        let w = Tensor.get weights [j; 0] in
        Tensor.set_ final_weights [idx; 0] (w *. group_weight)
      ) indices
    ) group_weights;
    
    final_weights
end

module RiskDecomposition = struct
  let portfolio_risk_decomposition ~weights ~sigma ~rho ~n =
    let n_float = float_of_int n in
    
    let sum_w = 
      Tensor.sum weights ~dim:[0] ~keepdim:false
      |> Tensor.to_float0_exn in
    let sum_w2 = 
      Tensor.pow weights 2.
      |> Tensor.sum ~dim:[0] ~keepdim:false
      |> Tensor.to_float0_exn in
    
    let total_risk = 
      sigma *. sqrt (
        sum_w2 *. (1. -. rho) +.
        sum_w *. sum_w *. rho
      ) in
    
    let systematic_risk =
      sigma *. sqrt (sum_w *. sum_w *. rho) in
    
    let residual_risk =
      sigma *. sqrt (sum_w2 *. (1. -. rho)) in
    
    let risk_ratio = residual_risk /. systematic_risk in
    
    {total_risk; systematic_risk; residual_risk; risk_ratio}

  let marginal_risk_contributions ~weights ~cov =
    let total_risk = 
      Tensor.sqrt (
        Tensor.mm (Tensor.mm (Tensor.transpose2 weights) cov) weights
      ) |> Tensor.to_float0_exn in
    
    let marginal_risks = 
      Tensor.mm cov weights
      |> Tensor.div_scalar total_risk in
    
    Array.init (Tensor.size weights 0) (fun i ->
      Tensor.get marginal_risks [i; 0]
    )

  let component_var ~weights ~cov ~confidence =
    let n = Tensor.size weights 0 in
    
    (* Compute portfolio volatility *)
    let vol = 
      Tensor.sqrt (
        Tensor.mm (Tensor.mm (Tensor.transpose2 weights) cov) weights
      ) |> Tensor.to_float0_exn in
    
    (* Compute normal quantile *)
    let z_score = sqrt 2. *. erfinv (2. *. confidence -. 1.) in
    
    (* Compute component VaR *)
    let marginal_risks = marginal_risk_contributions ~weights ~cov in
    Array.map (fun mr -> -1. *. mr *. z_score *. vol) marginal_risks
end

open Torch

module AsymptoticAnalysis = struct
  let compute_convergence_properties ~returns ~max_size =
    let size_effects = Array.make max_size (0, 0.) in
    let n_samples = 100 in
    
    (* Compute variances for different portfolio sizes *)
    for size = 2 to max_size do
      let variances = ref [] in
      for _ = 1 to n_samples do
        let indices = 
          Tensor.randperm (Tensor.size returns 1)
          |> Tensor.narrow 0 0 size in
        let subset = Tensor.index_select returns 1 indices in
        
        let weights = 
          Tensor.div_scalar (Tensor.ones [1; size]) 
            (float_of_int size) in
        let port_returns = Tensor.mm weights subset in
        let variance = 
          Tensor.var port_returns ~dim:[1] ~unbiased:true ~keepdim:false
          |> Tensor.to_float0_exn in
        
        variances := variance :: !variances
      done;
      
      let mean_var = 
        List.fold_left (+.) 0. !variances /. float_of_int n_samples in
      size_effects.(size-2) <- (size, mean_var)
    done;
    
    (* Compute convergence rates *)
    let theoretical_rate = -0.5 in  (* From theory *)
    
    let empirical_rate =
      let x = Array.map (fun (s, _) -> log (float_of_int s)) size_effects in
      let y = Array.map (fun (_, v) -> log v) size_effects in
      let n = Array.length x in
      
      let mean_x = Array.fold_left (+.) 0. x /. float_of_int n in
      let mean_y = Array.fold_left (+.) 0. y /. float_of_int n in
      
      let numerator = ref 0. in
      let denominator = ref 0. in
      for i = 0 to n-1 do
        numerator := !numerator +. 
          (x.(i) -. mean_x) *. (y.(i) -. mean_y);
        denominator := !denominator +. 
          (x.(i) -. mean_x) *. (x.(i) -. mean_x)
      done;
      !numerator /. !denominator in
    
    (* Find stable region *)
    let stable_start = ref 2 in
    let stable_end = ref max_size in
    let threshold = 0.1 in
    
    let is_stable i =
      if i < 2 || i >= max_size-1 then false
      else
        let (_, v_prev) = size_effects.(i-1) in
        let (_, v_curr) = size_effects.(i) in
        let (_, v_next) = size_effects.(i+1) in
        abs_float ((v_curr -. v_prev) /. v_prev) < threshold &&
        abs_float ((v_next -. v_curr) /. v_curr) < threshold in
    
    for i = 2 to max_size-2 do
      if is_stable i then begin
        stable_start := min !stable_start i;
        stable_end := max !stable_end i
      end
    done;
    
    (* Compute reliability metric *)
    let reliability =
      let expected_ratio = exp theoretical_rate in
      let actual_ratio = exp empirical_rate in
      1. -. abs_float (expected_ratio -. actual_ratio) /.
           abs_float expected_ratio in
    
    {
      theoretical_rate;
      empirical_rate;
      size_effects;
      stable_region = (!stable_start, !stable_end);
      reliability;
    }

  let analyze_limiting_behavior ~alpha ~sigma ~rho ~size_range =
    let compute_portfolio_stats n weights =
      let n_float = float_of_int n in
      
      (* Compute portfolio variance *)
      let variance = 
        let sum_w = 
          Tensor.sum weights ~dim:[0] ~keepdim:false
          |> Tensor.to_float0_exn in
        let sum_w2 = 
          Tensor.pow weights 2.
          |> Tensor.sum ~dim:[0] ~keepdim:false
          |> Tensor.to_float0_exn in
        
        sigma *. sigma *. (
          sum_w2 *. (1. -. rho) +.
          sum_w *. sum_w *. rho
        ) in
      
      (* Compute numerical stability *)
      let condition_number = 
        let cov = 
          let corr = 
            let ones = Tensor.ones [n; n] in
            let eye = Tensor.eye n in
            Tensor.add eye (Tensor.mul_scalar (Tensor.sub ones eye) rho) in
          Tensor.mul_scalar corr (sigma *. sigma) in
        
        let eigvals = 
          Tensor.linalg_eigvalsh cov ~UPLO:'L' in
        let max_eigval = 
          Tensor.max eigvals |> Tensor.to_float0_exn in
        let min_eigval = 
          Tensor.min eigvals |> Tensor.to_float0_exn in
        max_eigval /. min_eigval in
      
      (variance, condition_number < 1000.) in
    
    (* Analyze small-N regime *)
    let small_n = List.hd size_range in
    let small_n_weights = 
      let ones = Tensor.ones [small_n; 1] in
      let mean_alpha = 
        Tensor.mean alpha ~dim:[0] ~keepdim:true in
      let alpha_centered = 
        Tensor.sub alpha (Tensor.mm ones mean_alpha) in
      Tensor.div_scalar alpha_centered 
        (2. *. sigma *. sigma *. (1. -. rho)) in
    
    let (small_n_var, small_n_stable) = 
      compute_portfolio_stats small_n small_n_weights in
    
    (* Find transition region *)
    let transition_start = ref small_n in
    let transition_end = ref small_n in
    let prev_weights = ref small_n_weights in
    let convergence_rates = ref [] in
    
    List.iter (fun n ->
      if n > small_n then begin
        let weights = 
          let ones = Tensor.ones [n; 1] in
          let mean_alpha = 
            Tensor.mean alpha ~dim:[0] ~keepdim:true in
          let alpha_centered = 
            Tensor.sub alpha (Tensor.mm ones mean_alpha) in
          Tensor.div_scalar alpha_centered 
            (2. *. sigma *. sigma *. (1. -. rho)) in
        
        let weight_diff = 
          Tensor.sub weights !prev_weights
          |> Tensor.abs
          |> Tensor.mean
          |> Tensor.to_float0_exn in
        
        if weight_diff > 0.1 then
          transition_end := n;
        
        convergence_rates := 
          weight_diff /. sqrt (float_of_int n) :: !convergence_rates;
        
        prev_weights := weights
      end
    ) size_range;
    
    let convergence_rate = 
      List.fold_left (+.) 0. !convergence_rates /. 
      float_of_int (List.length !convergence_rates) in
    
    (* Analyze large-N regime *)
    let large_n = List.hd (List.rev size_range) in
    let limiting_weights = 
      let ones = Tensor.ones [large_n; 1] in
      let mean_alpha = 
        Tensor.mean alpha ~dim:[0] ~keepdim:true in
      let alpha_centered = 
        Tensor.sub alpha (Tensor.mm ones mean_alpha) in
      Tensor.div_scalar alpha_centered 
        (2. *. sigma *. sigma *. (1. -. rho)) in
    
    let (asymptotic_variance, numerical_stability) = 
      compute_portfolio_stats large_n limiting_weights in
    
    {
      small_n = {
        weights = small_n_weights;
        variance = small_n_var;
        stability = if small_n_stable then 1. else 0.;
      };
      transition = {
        start_size = !transition_start;
        end_size = !transition_end;
        convergence_rate;
      };
      large_n = {
        limiting_weights;
        asymptotic_variance;
        numerical_stability;
      };
    }

  let compute_asymptotic_properties ~rho ~sizes =
    let n = List.length sizes in
    let theoretical_dof = Array.make n 0. in
    let variance_ratios = Array.make n 0. in
    let stable_sizes = ref [] in
    
    List.iteri (fun i size ->
      let n_float = float_of_int size in
      
      theoretical_dof.(i) <- 
        n_float /. (1. +. (n_float -. 1.) *. rho);
      
      (* Variance ratio *)
      let systematic = rho *. (1. +. (n_float -. 1.) *. rho) in
      let residual = (1. -. rho) /. n_float in
      variance_ratios.(i) <- residual /. systematic;
      
      (* Check stability *)
      if i > 0 then
        if abs_float (theoretical_dof.(i) /. theoretical_dof.(i-1) -. 1.) < 0.01
        then stable_sizes := size :: !stable_sizes
    ) sizes;
    
    {
      theoretical_dof;
      variance_ratios;
      stable_sizes = List.rev !stable_sizes;
    }
end