open Torch

type timestamp = float
type quantile = float

type dataset = {
  y: Tensor.t;       (* Target variable *)
  x: Tensor.t;       (* Covariates matrix *)
  times: timestamp array;
}

type density_estimate = {
  pdf: Tensor.t;
  cdf: Tensor.t;
  support: Tensor.t;
  bandwidth: float;
}

type stationarity_test = {
  is_stationary: bool;
  test_statistic: float;
  critical_values: float array;
  p_value: float;
}

type ergodicity_test = {
  is_ergodic: bool;
  mixing_coefficient: float;
  convergence_rate: float;
}

type model_config = {
  dimension: int;
  num_quantiles: int;
  num_lags: int;
  quantile_levels: float array;
  max_iterations: int;
  convergence_tol: float;
}

type parameters = {
  beta: Tensor.t;     (* Covariate coefficients *)
  theta: Tensor.t;    (* Autoregressive coefficients *)
  lambda: float;      (* Crossing penalty parameter *)
}

type model_state = {
  quantiles: Tensor.t;      (* Current quantile estimates *)
  quantile_history: Tensor.t;  (* Historical quantile values *)
  volatility: Tensor.t;     (* Volatility estimates *)
  inertia: float;          (* Quantile stickiness parameter *)
}

type divergence_components = {
  pinball: float;
  crossing: float;
  volatility: float;
  total: float;
}

(* Stationarity module *)
module Stationarity = struct
  (* Compute rolling statistics *)
  let compute_rolling_stats tensor window_size =
    let n = Tensor.size tensor 0 in
    let means = Array.make (n - window_size + 1) (Tensor.zeros []) in
    let vars = Array.make (n - window_size + 1) (Tensor.zeros []) in
    for i = 0 to n - window_size do
      let window = Tensor.narrow tensor 0 i window_size in
      means.(i) <- Tensor.mean window;
      vars.(i) <- Tensor.var window
    done;
    means, vars

  (* Compute autocovariance *)
  let compute_autocovariance tensor max_lag =
    let n = size tensor 0 in
    let centered = sub tensor (mean tensor) in
    Array.init max_lag (fun lag ->
      let y1 = narrow centered 0 0 (n - lag) in
      let y2 = narrow centered 0 lag (n - lag) in
      to_float0_exn (mean (mul y1 y2))
    )

  (* KPSS test *)
  let kpss_test tensor =
    let n = size tensor 0 in
    let partial_sums = cumsum tensor 0 in
    let variance = var tensor in
    
    (* Compute long-run variance estimate *)
    let max_lag = min 20 (n / 4) in
    let auto_cov = compute_autocovariance tensor max_lag in
    let bartlett_weights = Array.init max_lag (fun i ->
      1. -. float i /. (float max_lag +. 1.)
    ) in
    let long_run_var = variance +.
      2. *. Array.fold_left2 (fun acc w ac -> acc +. w *. ac)
                            0. bartlett_weights auto_cov in
    
    (* Compute test statistic *)
    let stat = to_float0_exn (sum (pow partial_sums 2)) /.
               (float n *. float n *. long_run_var) in
    
    (* Critical values for 1%, 5%, 10% significance *)
    let critical_values = [|0.739; 0.463; 0.347|] in
    let p_value = 0.05 in
    
    {
      is_stationary = stat < critical_values.(1);
      test_statistic = stat;
      critical_values;
      p_value;
    }

  (* Augmented Dickey-Fuller test *)
  let adf_test tensor lags =
    let n = size tensor 0 in
    let y = narrow tensor 0 1 (n - 1) in
    let x = narrow tensor 0 0 (n - 1) in
    let diff = sub y x in
    
    (* Create lagged differences *)
    let lagged_diffs = Array.init lags (fun i ->
      narrow tensor 0 i (n - lags - 1)
    ) in
    
    (* Construct regression matrix *)
    let x_mat = cat [x; stack lagged_diffs 0] 1 in
    
    (* OLS estimation *)
    let beta = matmul (pinverse x_mat) diff in
    let residuals = sub diff (matmul x_mat beta) in
    let std_err = sqrt (var residuals) in
    
    (* Test statistic *)
    let test_stat = to_float0_exn (div (get beta [0]) std_err) in
    let critical_values = [|-3.43; -2.86; -2.57|] in
    
    {
      is_stationary = test_stat < critical_values.(1);
      test_statistic = test_stat;
      critical_values;
      p_value = 0.05;  (* Approximation *)
    }
end

(* Ergodicity module *)
module Ergodicity = struct
  (* Estimate mixing coefficients *)
  let estimate_mixing_coefficient tensor window_size =
    let n = size tensor 0 in
    let max_lag = min (n/4) 50 in
    
    let compute_dependence lag =
      let y1 = narrow tensor 0 0 (n - lag) in
      let y2 = narrow tensor 0 lag (n - lag) in
      let joint_mean = mean (mul y1 y2) in
      let prod_means = mul (mean y1) (mean y2) in
      abs (sub joint_mean prod_means)
    in
    
    let mixing_coefs = Array.init max_lag compute_dependence in
    let alpha = Array.fold_left (fun acc x -> 
      acc +. Tensor.to_float0_exn x) 0. mixing_coefs in
    
    alpha /. float max_lag

  (* Test ergodicity *)
  let test_ergodicity tensor =
    let n = size tensor 0 in
    let window_size = max 30 (n / 10) in
    let mixing_coef = estimate_mixing_coefficient tensor window_size in
    
    (* Compute convergence rate *)
    let partial_means = Array.init (n - window_size) (fun i ->
      let window = narrow tensor 0 i window_size in
      to_float0_exn (mean window)
    ) in
    let diffs = Array.init (Array.length partial_means - 1) (fun i ->
      abs_float (partial_means.(i+1) -. partial_means.(i))
    ) in
    let conv_rate = Array.fold_left (+.) 0. diffs /. float (Array.length diffs) in
    
    {
      is_ergodic = mixing_coef < 0.5;  (* Threshold *)
      mixing_coefficient = mixing_coef;
      convergence_rate = conv_rate;
    }
end

(* Density estimation module *)
module Density = struct
  (* Kernel functions *)
  let gaussian_kernel x =
    div (exp (neg (div (pow x 2) 2.)))
        (sqrt (mul_scalar (ones [1]) (2. *. Float.pi)))

  let epanechnikov_kernel x =
    where' (le (abs x) 1.)
           (sub 1. (pow x 2))
           (zeros_like x)

  (* Silverman's rule for bandwidth selection *)
  let silverman_bandwidth data =
    let n = float (size data 0) in
    let std = to_float0_exn (std data) in
    let iqr = to_float0_exn (
      sub (quantile data 0.75 ~dim:0)
          (quantile data 0.25 ~dim:0)
    ) in
    let scale = min std (iqr /. 1.34) in
    0.9 *. scale *. (n ** (-0.2))

  (* Boundary correction using reflection method *)
  let boundary_correction x min_x max_x kernel =
    let base = kernel x in
    let reflected_left = where' (le x min_x) (kernel (sub min_x x)) (zeros_like x) in
    let reflected_right = where' (ge x max_x) (kernel (sub x max_x)) (zeros_like x) in
    add base (add reflected_left reflected_right)

  (* Kernel density estimation with boundary correction *)
  let estimate_density ?(n_points=200) data =
    let n = size data 0 in
    let min_val = to_float0_exn (min data) in
    let max_val = to_float0_exn (max data) in
    let h = silverman_bandwidth data in
    
    let support = linspace ~start:min_val ~end_:max_val n_points in
    let pdf = zeros [n_points] in
    
    for i = 0 to n_points - 1 do
      let x = get support [i] in
      let z = div (sub data x) h in
      let k = boundary_correction z min_val max_val gaussian_kernel in
      let density = div (sum k) (float n *. h) in
      set_ pdf [i] density
    done;
    
    (* Compute CDF through numerical integration *)
    let cdf = cumsum pdf 0 in
    let norm_const = to_float0_exn (sum pdf) in
    
    {
      pdf;
      cdf = div cdf norm_const;
      support;
      bandwidth = h;
    }
end

(* Memory-efficient tensor operations *)
module TensorOps = struct
  let efficient_matmul a b ~out =
    match out with
    | Some o -> Tensor.matmul_out o a b
    | None -> Tensor.matmul a b

  let update_inplace dest src =
    Tensor.copy_ dest src;
    dest

  let clean_computation f x =
    let result = f x in
    Gc.minor ();
    result
end

(* Core CAViaR dynamics *)
module Dynamics = struct
  (* Multi-lag structure *)
  module MultiLag = struct
    let compute_lagged_effects tensor lags =
      let n = size tensor 0 in
      let max_lag = Array.fold_left max 0 lags in
      Array.map (fun lag ->
        let padded = constant_pad_nd tensor [lag; 0] 0. in
        narrow padded 0 0 n
      ) lags

    let aggregate_lag_effects lagged_values coeffs =
      Array.fold_left2 (fun acc lag coef ->
        add acc (mul lag coef)
      ) (zeros_like (Array.get lagged_values 0)) lagged_values coeffs
  end

  (* Volatility clustering *)
  module Volatility = struct
    let estimate_local_volatility data window_size =
      let n = size data 0 in
      let vol = zeros [n] in
      
      for i = window_size to n - 1 do
        let window = narrow data 0 (i - window_size) window_size in
        let local_std = std window in
        set_ vol [i] local_std
      done;
      vol

    let compute_volatility_impact data volatility params =
      let scaled_vol = mul volatility params.theta in
      add data scaled_vol
  end

  (* Quantile evolution with inertia *)
  let evolve_quantiles params state data =
    (* Linear combination of covariates *)
    let location = matmul params.beta data in
    
    (* Compute autoregressive effects *)
    let lagged_effects = MultiLag.compute_lagged_effects 
                          state.quantile_history 
                          [|1; 2; 5; 10|] in
    let ar_component = MultiLag.aggregate_lag_effects 
                        lagged_effects 
                        (Array.init 4 (fun i -> 
                           get params.theta [i])) in
    
    (* Update with volatility clustering *)
    let volatility_impact = Volatility.compute_volatility_impact
                            location state.volatility params in
    
    (* Combine with inertia *)
    let raw_quantiles = add volatility_impact ar_component in
    mul_scalar raw_quantiles (1. -. state.inertia)
end

(* Non-crossing constraints and ordering *)
module Constraints = struct
  let compute_crossing_measure q1 q2 =
    relu (sub q1 q2)

  let enforce_ordering quantiles tau_levels =
    let n = size quantiles 0 in
    let sorted = sort quantiles 0 |> fst in
    let needs_reorder = ref false in
    
    for i = 0 to n - 2 do
      if tau_levels.(i) > tau_levels.(i + 1) then
        needs_reorder := true
    done;
    
    if !needs_reorder then
      let ordered = copy quantiles in
      for i = 1 to n - 1 do
        let prev = get sorted [i-1] in
        let curr = get sorted [i] in
        let alpha = (tau_levels.(i) -. tau_levels.(i-1)) /. 
                   (tau_levels.(n-1) -. tau_levels.(0)) in
        let interp = add (mul_scalar prev (1. -. alpha))
                        (mul_scalar curr alpha) in
        update_inplace (select ordered [i]) interp
      done;
      ordered
    else
      quantiles

  let compute_total_crossing quantiles =
    let n = size quantiles 0 in
    let total = ref (zeros []) in
    for i = 0 to n - 2 do
      let q1 = select quantiles [i] in
      let q2 = select quantiles [i + 1] in
      total := add !total (compute_crossing_measure q1 q2)
    done;
    !total
end

(* Divergence computation *)
module Divergence = struct
  (* Pinball divergence *)
  let pinball_divergence tau y y_pred =
    let diff = sub y y_pred in
    let pos_part = mul_scalar diff tau in
    let neg_part = mul_scalar diff (tau -. 1.) in
    mean (where' diff pos_part neg_part)

  (* Multi-quantile CAViaR divergence *)
  let compute_divergence params state data config =
    (* Compute quantile predictions *)
    let predictions = Dynamics.evolve_quantiles params state data in
    let ordered_pred = Constraints.enforce_ordering predictions 
                                                  config.quantile_levels in
    
    (* Compute base pinball divergences *)
    let quantile_divergences = Array.init config.num_quantiles (fun i ->
      let pred_i = select ordered_pred [i] in
      let tau_i = config.quantile_levels.(i) in
      pinball_divergence tau_i data pred_i
    ) in
    let base_divergence = sum (stack quantile_divergences 0) in
    
    (* Compute crossing penalty *)
    let crossing = Constraints.compute_total_crossing ordered_pred in
    let crossing_penalty = mul_scalar crossing params.lambda in
    
    (* Compute volatility impact *)
    let vol_impact = sum (mul state.volatility params.theta) in
    
    {
      pinball = to_float0_exn base_divergence;
      crossing = to_float0_exn crossing_penalty;
      volatility = to_float0_exn vol_impact;
      total = to_float0_exn (add (add base_divergence crossing_penalty) vol_impact)
    }
end

(* Parameter adaptation *)
module Adaptation = struct
  let adapt_parameters params divergence_history =
    let n = Array.length divergence_history in
    if n < 10 then params
    else
      let recent_divergences = Array.sub divergence_history (n-10) 10 in
      let avg_divergence = Array.fold_left (+.) 0. recent_divergences /. 10. in
      
      (* Adapt lambda based on crossing frequency *)
      let new_lambda = if avg_divergence > divergence_history.(n-1) then
        params.lambda *. 1.1
      else
        max 1.0 (params.lambda *. 0.95) in
      
      { params with lambda = new_lambda }

  let update_state state predictions data window_size =
    (* Update quantile history *)
    let new_history = cat [narrow state.quantile_history 0 1 
                                 (size state.quantile_history 0 - 1);
                          unsqueeze predictions 0] 0 in
    
    (* Update volatility estimate *)
    let new_volatility = Dynamics.Volatility.estimate_local_volatility 
                          data window_size in
    
    (* Adapt inertia based on prediction accuracy *)
    let pred_error = abs (sub data predictions) in
    let avg_error = mean pred_error in
    let new_inertia = min 0.9 (state.inertia *. 
                              (1. +. to_float0_exn avg_error)) in
    
    { state with
      quantile_history = new_history;
      volatility = new_volatility;
      inertia = new_inertia }
end

module CMAES = struct
  (* Core strategy parameters and state *)
  type strategy_params = {
    mu: int;                    (* Number of parents *)
    weights: Tensor.t;          (* Selection weights *)
    mueff: float;              (* Variance-effective selection mass *)
    cc: float;                 (* Cumulation factor for C *)
    cs: float;                 (* Cumulation factor for sigma *)
    c1: float;                 (* Learning rate for rank-one update *)
    cmu: float;                (* Learning rate for rank-mu update *)
    damps: float;              (* Damping for sigma *)
  }

  type state = {
    dimension: int;
    population_size: int;
    params: strategy_params;
    mean: Tensor.t;            (* Distribution mean *)
    sigma: float;              (* Step size *)
    pc: Tensor.t;              (* Evolution path for C *)
    ps: Tensor.t;              (* Evolution path for sigma *)
    C: Tensor.t;               (* Covariance matrix *)
    B: Tensor.t;              (* Eigenvectors of C *)
    D: Tensor.t;              (* Eigenvalues of D *)
    eigeneval: int;           (* B and D updated at *)
    generation: int;
    restarts: int;
    stagnation_count: int;
  }

  (* Strategy parameter initialization *)
  let compute_strategy_params dimension population_size =
    let mu = population_size / 4 in
    let weights = Tensor.init [mu] (fun i ->
      log ((float population_size +. 1.) /. 2.) -. log (float (i + 1))
    ) in
    let normalized_weights = Tensor.div weights (Tensor.sum weights) in
    let mueff = 1. /. Tensor.(to_float0_exn (sum (pow normalized_weights 2.))) in
    
    {
      mu;
      weights = normalized_weights;
      mueff;
      cc = 4. /. float dimension;
      cs = (mueff +. 2.) /. (float dimension +. mueff +. 5.);
      c1 = 2. /. ((float dimension +. 1.3) ** 2. +. mueff);
      cmu = min 1. -.
            (2. *. (mueff -. 2. +. 1. /. mueff) /. 
             ((float dimension +. 2.) ** 2. +. mueff));
      damps = 1. +. 2. *. max 0. (sqrt ((mueff -. 1.) /. 
                                       (float dimension +. 1.)) -. 1.);
    }

  (* Initialization *)
  let create dimension ~population_size =
    let params = compute_strategy_params dimension population_size in
    {
      dimension;
      population_size;
      params;
      mean = Tensor.zeros [dimension];
      sigma = 1.0;
      pc = Tensor.zeros [dimension];
      ps = Tensor.zeros [dimension];
      C = Tensor.eye dimension;
      B = Tensor.eye dimension;
      D = Tensor.ones [dimension];
      eigeneval = 0;
      generation = 0;
      restarts = 0;
      stagnation_count = 0;
    }

  (* Eigendecomposition handling *)
  module Decomposition = struct
    let update_eigensystem state =
      let eig = symeig state.C ~eigenvectors:true in
      let eigvals, eigvecs = fst eig, snd eig in
      { state with
        B = eigvecs;
        D = sqrt eigvals;
        eigeneval = state.generation }

    let should_update state =
      state.generation >= state.eigeneval +
      int_of_float (1.0 /. (state.params.c1 +. state.params.cmu) /. 
                   float state.dimension /. 10.0)

    let condition_number state =

      to_float0_exn (div (max state.D) (min state.D))
  end

  (* Population generation and sampling *)
  module Sampling = struct
    let generate_population state =
      if Decomposition.should_update state then
        let state = Decomposition.update_eigensystem state in
        let z = randn [state.population_size; state.dimension] in
        let x = add state.mean
          (mul_scalar (matmul z (mul state.B (diag state.D)))
                     state.sigma) in
        x, state
      else
        let z = randn [state.population_size; state.dimension] in
        let x = add state.mean
          (mul_scalar (matmul z (mul state.B (diag state.D)))
                     state.sigma) in
        x, state

    let sample_single state =
      let z = randn [state.dimension] in
      add state.mean
          (mul_scalar (mv (mul state.B (diag state.D)) z)
                     state.sigma)
  end

  (* Evolution path updates *)
  module Evolution = struct
    let update_evolution_paths state y z =
      (* Update ps (conjugate evolution path) *)
      let ps_new = add (mul_scalar state.ps (1. -. state.params.cs))
                      (mul_scalar z 
                         (sqrt (state.params.cs *. 
                               (2. -. state.params.cs)))) in
      
      (* Compute h_sig (heaviside function) *)
      let hsig = to_float0_exn (norm ps_new) /. 
                 sqrt (1. -. (1. -. state.params.cs) ** 
                           (2. *. float state.generation))
                 <= (1.4 +. 2. /. (float state.dimension +. 1.)) in
      
      (* Update pc (evolution path) *)
      let pc_new = add (mul_scalar state.pc (1. -. state.params.cc))
                      (if hsig then 
                         mul_scalar y (sqrt (state.params.cc *. 
                                           (2. -. state.params.cc)))
                       else zeros_like state.pc) in
      
      ps_new, pc_new, hsig
  end

  (* Covariance matrix adaptation *)
  module Adaptation = struct
    let update_covariance state ~population ~fitness =
      let sorted_idx = argsort fitness in
      let selected_idx = narrow sorted_idx 0 0 state.params.mu in
      let selected = index_select population 0 selected_idx in
      
      (* Update mean *)
      let old_mean = state.mean in
      let new_mean = sum (mul state.params.weights selected) in
      
      (* Compute evolution parameters *)
      let y = div (sub new_mean old_mean) state.sigma in
      let z = div y (mul state.B (diag state.D)) in
      
      (* Update evolution paths *)
      let ps_new, pc_new, hsig = Evolution.update_evolution_paths state y z in
      
      (* Update step size *)
      let sigma_new = state.sigma *. 
                     exp ((norm ps_new /. norm state.ps -. 1.) /. 3.) in
      
      (* Update covariance matrix *)
      let c1a = state.params.c1 *. 
                (1. -. (1. -. (if hsig then 1. else 0.) ** 2.) *. 
                 state.params.cc *. (2. -. state.params.cc)) in
      
      let rank_one = mul (reshape pc_new [-1; 1])
                        (reshape pc_new [1; -1]) in
      let rank_mu = matmul (transpose selected 0 1) selected in
      
      let C_new = add (mul_scalar state.C (1. -. c1a -. state.params.cmu))
                      (add (mul_scalar rank_one c1a)
                           (mul_scalar rank_mu 
                             (state.params.cmu /. (sigma_new ** 2.)))) in
      
      { state with
        mean = new_mean;
        sigma = sigma_new;
        pc = pc_new;
        ps = ps_new;
        C = C_new;
        generation = state.generation + 1 }
  end

  (* Restart strategies *)
  module Restart = struct
    type restart_config = {
      max_restarts: int;
      stagnation_tolerance: float;
      improvement_threshold: float;
      increase_popsize: bool;
    }

    let should_restart state config fitness_history =
      let n = Array.length fitness_history in
      if n < 10 then false
      else
        let recent_improvement = 
          abs_float (fitness_history.(n-1) -. fitness_history.(n-10)) in
        recent_improvement < config.stagnation_tolerance ||
        state.stagnation_count > 20

    let increase_population_size state =
      let new_popsize = state.population_size * 2 in
      let new_params = compute_strategy_params state.dimension new_popsize in
      { state with
        population_size = new_popsize;
        params = new_params }

    let reinitialize state ~increase_popsize =
      let state = if increase_popsize then 
        increase_population_size state else state in
      { state with
        sigma = state.sigma *. 2.;
        pc = Tensor.zeros_like state.pc;
        ps = Tensor.zeros_like state.ps;
        C = Tensor.eye state.dimension;
        B = Tensor.eye state.dimension;
        D = Tensor.ones [state.dimension];
        eigeneval = 0;
        generation = 0;
        stagnation_count = 0;
        restarts = state.restarts + 1 }
  end
end

(* Optimization *)
module Optimization = struct
  type optimization_state = {
    best_solution: Tensor.t;
    best_fitness: float;
    current_state: state;
    history: float array;
  }

  let create_optimizer dimension ~population_size =
    let cmaes = create dimension ~population_size in
    {
      best_solution = Tensor.zeros [dimension];
      best_fitness = Float.infinity;
      current_state = cmaes;
      history = [||];
    }

  let step state ~objective =
    let population, cmaes_state = Sampling.generate_population state.current_state in
    
    (* Evaluate population *)
    let fitness = Array.init state.current_state.population_size (fun i ->
      objective (Tensor.select population 0 i)
    ) in
    
    (* Track best solution *)
    let min_idx = ref 0 in
    let min_fit = ref fitness.(0) in
    for i = 1 to Array.length fitness - 1 do
      if fitness.(i) < !min_fit then begin
        min_fit := fitness.(i);
        min_idx := i
      end
    done;
    
    let state = 
      if !min_fit < state.best_fitness then
        { state with
          best_solution = Tensor.select population 0 !min_idx;
          best_fitness = !min_fit;
          current_state = { cmaes_state with stagnation_count = 0 } }
      else
        { state with
          current_state = { cmaes_state with 
                           stagnation_count = 
                             cmaes_state.stagnation_count + 1 } }
    in
    
    (* Update state *)
    let updated_state = Adaptation.update_covariance 
      state.current_state
      ~population
      ~fitness:(Tensor.of_float1 fitness) in
    
    { state with
      current_state = updated_state;
      history = Array.append state.history [|!min_fit|] }

  let optimize ~objective ~init_state ~config =
    let rec optimization_loop state =
      if state.current_state.generation >= config.max_iterations ||
         state.current_state.restarts >= config.max_restarts then
        state
      else begin
        let state = step state ~objective in
        
        if Restart.should_restart state.current_state config state.history then
          let new_cmaes = Restart.reinitialize state.current_state 
                           ~increase_popsize:config.increase_popsize in
          optimization_loop { state with current_state = new_cmaes }
        else
          optimization_loop state
      end
    in
    optimization_loop init_state
end