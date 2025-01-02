open Torch

type state = {
  a: float;
  b: float;
}

type observation = {
  y: float;
  v: float;
  mu: float;
}

module StateSpace = struct
  type t = state

  let init ~a ~b = {a; b}

  let validate state =
    state.a > 0.0 && state.b > 0.0

  let mean state = state.b /. state.a

  let variance state =
    if state.a <= 1.0 then infinity
    else 1.0 /. (state.a -. 1.0)

  let update state obs =
    {
      a = state.a +. obs.v;
      b = state.b +. obs.y /. obs.mu;
    }
end

module BetaPrime = struct
    type params = {
      alpha: float;
      beta: float;
      scale: float;
    }

    let pdf ~params ~x =
      if x <= 0.0 then 0.0
      else
        let scaled_x = x /. params.scale in
        let b = Stdlib.Float.beta params.alpha params.beta in
        (scaled_x ** (params.alpha -. 1.0)) *. 
        ((1.0 +. scaled_x) ** (-.(params.alpha +. params.beta))) /.
        (b *. params.scale)

    let log_pdf ~params ~x =
      if x <= 0.0 then neg_infinity
      else
        let scaled_x = x /. params.scale in
        let log_b = log (Stdlib.Float.beta params.alpha params.beta) in
        (params.alpha -. 1.0) *. log scaled_x -.
        (params.alpha +. params.beta) *. log (1.0 +. scaled_x) -.
        log_b -. log params.scale

    let cdf ~params ~x =
      if x <= 0.0 then 0.0
      else
        let n = 100 in  (* Number of quadrature points *)
        let h = x /. float_of_int n in
        let sum = ref 0.0 in
        for i = 0 to n - 1 do
          let x_i = float_of_int i *. h in
          let x_ip1 = x_i +. h in
          sum := !sum +. (pdf ~params ~x:x_i +. 
                         pdf ~params ~x:x_ip1) *. h /. 2.0
        done;
        !sum

    let generate ~params ~rng =
      let rec generate_aux () =
        let u = Random.State.float rng 1.0 in
        let x = Random.State.float rng 10.0 *. params.scale in
        let y = Random.State.float rng 1.0 in
        if y <= pdf ~params ~x then x
        else generate_aux ()
      in
      generate_aux ()
end

module PearsonVI = struct
    type params = {
      a: float;
      b: float;
      scale: float;
    }

    let from_beta_prime bp_params = {
      a = bp_params.BetaPrime.alpha;
      b = bp_params.BetaPrime.beta;
      scale = bp_params.BetaPrime.scale;
    }

    let pdf ~params ~x =
      let bp_params = BetaPrime.{
        alpha = params.a;
        beta = params.b;
        scale = params.scale;
      } in
      BetaPrime.pdf ~params:bp_params ~x

    let log_pdf ~params ~x =
      let bp_params = BetaPrime.{
        alpha = params.a;
        beta = params.b;
        scale = params.scale;
      } in
      BetaPrime.log_pdf ~params:bp_params ~x

    let mgf ~params ~t =
      if t >= params.b then infinity
      else
        let num = Stdlib.Float.gamma (params.a +. t) *. 
                 Stdlib.Float.gamma params.b in
        let den = Stdlib.Float.gamma params.a *. 
                 Stdlib.Float.gamma (params.b -. t) in
        num /. den *. (params.scale ** t)

    let moment ~params ~k =
      if k >= params.b then infinity
      else
        let num = Stdlib.Float.gamma (params.a +. float_of_int k) *. 
                 Stdlib.Float.gamma params.b in
        let den = Stdlib.Float.gamma params.a *. 
                 Stdlib.Float.gamma (params.b -. float_of_int k) in
        num /. den *. (params.scale ** float_of_int k)
end

module StatisticalValidation = struct
  type validation_result = {
    valid: bool;
    error_msg: string option;
    test_statistics: float array;
  }

  let validate_parameters params =
    let stats = [|
      (if params.psi > 0.0 then 1.0 else 0.0);
      (if params.a_init > 1.0 then 1.0 else 0.0);
      (if params.gamma > 0.0 && params.gamma <= 1.0 then 1.0 else 0.0);
    |] in
    let valid = Array.for_all (fun x -> x > 0.0) stats in
    {
      valid;
      error_msg = if valid then None else Some "Invalid parameters";
      test_statistics = stats;
    }

  let validate_state_sequence states =
    let n = Array.length states in
    let stats = Array.make n 0.0 in
    for i = 0 to n - 1 do
      stats.(i) <- if StateSpace.validate states.(i) then 1.0 else 0.0
    done;
    let valid = Array.for_all (fun x -> x > 0.0) stats in
    {
      valid;
      error_msg = if valid then None else Some "Invalid state sequence";
      test_statistics = stats;
    }

  let validate_observations obs =
    let n = Array.length obs in
    let stats = Array.make n 0.0 in
    for i = 0 to n - 1 do
      stats.(i) <- if obs.(i).mu > 0.0 && obs.(i).v >= 0.0 then 1.0 else 0.0
    done;
    let valid = Array.for_all (fun x -> x > 0.0) stats in
    {
      valid;
      error_msg = if valid then None else Some "Invalid observations";
      test_statistics = stats;
    }

  let validate_model_fit params obs =
    let param_valid = validate_parameters params in
    let obs_valid = validate_observations obs in
    {
      valid = param_valid.valid && obs_valid.valid;
      error_msg = if param_valid.valid && obs_valid.valid then None 
                 else Some "Model validation failed";
      test_statistics = Array.append 
        param_valid.test_statistics 
        obs_valid.test_statistics;
    }
end

module EdgeCases = struct
  type edge_case_type =
    | ZeroExposure
    | InfiniteMean
    | ZeroVariance
    | DegenerateState
    | BoundaryCondition

  let detect_edge_cases state obs =
    if obs.v = 0.0 then Some ZeroExposure
    else if state.a <= 1.0 then Some InfiniteMean
    else if state.b = 0.0 then Some ZeroVariance
    else if not (StateSpace.validate state) then Some DegenerateState
    else if abs_float (state.b /. state.a -. 1.0) < 1e-10 then Some BoundaryCondition
    else None

  let handle_edge_case case state obs =
    match case with
    | ZeroExposure -> state
    | InfiniteMean -> {a = max 1.1 state.a; b = state.b}
    | ZeroVariance -> {state with b = max 1e-10 state.b}
    | DegenerateState -> StateSpace.init ~a:2.0 ~b:2.0
    | BoundaryCondition -> {state with b = state.b *. 1.01}

  let validate_edge_case_handling states =
    Array.for_all StateSpace.validate states
end

module TimeSeriesAnalysis = struct
  let compute_acf series max_lag =
    let n = Array.length series in
    let mean = Array.fold_left (+.) 0.0 series /. float_of_int n in
    let var = Array.fold_left (fun acc x ->
      let d = x -. mean in
      acc +. d *. d
    ) 0.0 series /. float_of_int n in
    
    Array.init max_lag (fun lag ->
      let sum = ref 0.0 in
      let n_t = n - lag in
      for t = 0 to n_t - 1 do
        let d1 = series.(t) -. mean in
        let d2 = series.(t + lag) -. mean in
        sum := !sum +. d1 *. d2
      done;
      !sum /. (float_of_int n_t *. var)
    )

  let ljung_box_test series max_lag =
    let n = float_of_int (Array.length series) in
    let acf = compute_acf series max_lag in
    
    let q_stat = Array.fold_left (fun acc rho_k ->
      acc +. (rho_k *. rho_k) /. (n -. float_of_int max_lag)
    ) 0.0 acf in
    
    let stat = n *. (n +. 2.0) *. q_stat in
    stat, 1.0 -. exp (-0.5 *. stat)  (* Approximate p-value *)
end

module DistributionTests = struct
  let ks_test data cdf =
    let n = Array.length data in
    let sorted = Array.copy data in
    Array.sort compare sorted;
    
    let max_diff = ref 0.0 in
    for i = 0 to n - 1 do
      let empirical = float_of_int (i + 1) /. float_of_int n in
      let theoretical = cdf sorted.(i) in
      max_diff := max !max_diff (abs_float (empirical -. theoretical))
    done;
    
    let stat = !max_diff *. sqrt (float_of_int n) in
    let critical_value = 1.36 /. sqrt (float_of_int n) in
    stat < critical_value

  let anderson_darling_test data cdf =
    let n = Array.length data in
    let sorted = Array.copy data in
    Array.sort compare sorted;
    
    let sum = ref 0.0 in
    for i = 0 to n - 1 do
      let z = cdf sorted.(i) in
      let term = float_of_int (2 * i + 1) in
      sum := !sum +. (term /. float_of_int n) *. 
             (log z +. log (1.0 -. cdf sorted.(n - 1 - i)))
    done;
    
    let a_squared = -. float_of_int n -. !sum in
    let significance = match a_squared with
      | x when x < 0.787 -> 0.25
      | x when x < 0.918 -> 0.15
      | x when x < 1.092 -> 0.10
      | x when x < 1.320 -> 0.05
      | _ -> 0.01
    in
    a_squared, significance
end

module RobustnessAnalysis = struct
  type robustness_metrics = {
    influence_measures: float array;
    leverage_points: int array;
    cook_distances: float array;
  }

  let compute_influence_measures params observations =
    let n = Array.length observations in
    let influence = Array.make n 0.0 in
    let leverage = Array.make n 0 in
    let cook = Array.make n 0.0 in
    
    (* Compute base model statistics *)
    let base_state = StateSpace.init ~a:params.a_init ~b:params.a_init in
    
    (* Compute influence measures *)
    for i = 0 to n - 1 do
      let reduced_obs = Array.concat [
        Array.sub observations 0 i;
        Array.sub observations (i + 1) (n - i - 1)
      ] in
      
      let state_i = StateSpace.update base_state observations.(i) in
      influence.(i) <- abs_float (
        StateSpace.mean state_i -. StateSpace.mean base_state
      );
      
      leverage.(i) <- if influence.(i) > 2.0 *. 
        (Array.fold_left (+.) 0.0 influence) /. float_of_int n 
        then 1 else 0;
      
      cook.(i) <- influence.(i) *. influence.(i) /. float_of_int n;
    done;
    
    {
      influence_measures = influence;
      leverage_points = leverage;
      cook_distances = cook;
    }
end

module SmithMillerComplete = struct
  type model_params = {
    psi: float;
    a_init: float;
    gamma: float;
  }

  module Init = struct
    let validate_init_params ~a_init ~psi ~gamma =
      if a_init <= 1.0 then
        invalid_arg "a_init must be greater than 1";
      if psi <= 0.0 then
        invalid_arg "psi must be positive";
      if gamma <= 0.0 || gamma > 1.0 then
        invalid_arg "gamma must be in (0,1]";
      {psi; a_init; gamma}

    let init_state params =
      let state = StateSpace.{
        a = params.a_init;
        b = params.a_init;
      } in
      if abs_float (StateSpace.mean state -. 1.0) > 1e-10 then
        invalid_arg "Initial state must have unit mean";
      state
  end

  module Observation = struct
    type individual_claim = {
      z: float;
      theta: float;
    }

    let process_observation params state obs =
      if obs.v = 0.0 then 0.0, state
      else begin
        let rng = Random.State.make_self_init () in
        let theta = state.StateSpace.a in
        let shape = obs.v /. params.psi in
        let scale = theta /. (obs.mu *. params.psi) in
        let y = Stdlib.Float.gamma shape /. scale in
        let new_state = StateSpace.{
          a = state.a +. obs.v /. params.psi;
          b = state.b +. y /. (obs.mu *. params.psi)
        } in
        y, new_state
      end

    let conditional_moments obs theta params =
      if obs.v = 0.0 then 0.0, 0.0
      else begin
        let mean = obs.v *. obs.mu /. theta in
        let var = obs.v *. params.psi *. obs.mu *. obs.mu /. (theta *. theta) in
        mean, var
      end
  end

  module Variance = struct
    let verify_variance_behavior state_seq gamma =
      let n = Array.length state_seq in
      if n < 2 then true
      else Array.fold_left (fun acc state ->
        acc && state.StateSpace.a > 1.0
      ) true state_seq

    let compute_variance_ratios state_seq =
      let n = Array.length state_seq in
      if n < 2 then [||]
      else Array.init (n-1) (fun i ->
        let v_curr = 1.0 /. (state_seq.(i).StateSpace.a -. 1.0) in
        let v_next = 1.0 /. (state_seq.(i+1).StateSpace.a -. 1.0) in
        v_next /. v_curr
      )

    let analyze_variance_stability state_seq gamma =
      let ratios = compute_variance_ratios state_seq in
      let theoretical_ratio = 1.0 /. gamma in
      Array.map (fun ratio ->
        abs_float (ratio -. theoretical_ratio)
      ) ratios
  end

  let create = Init.validate_init_params

  let filter params observations =
    let state = ref (Init.init_state params) in
    Array.map (fun obs ->
      let y, new_state = Observation.process_observation params !state obs in
      state := new_state;
      y, !state
    ) observations
end

module GeneralizedSM = struct
  type model_params = {
    psi: float;
    a_init: float;
    xi: float array;
    convex_space: bool;
  }

  module ParameterSpace = struct
    type xi_params = {
      values: float array;
      bounds: (float * float) array;
    }

    let validate_xi params =
      if Array.length params.values <> 6 then
        invalid_arg "Xi must have exactly 6 components";
      if params.values |> Array.exists (fun x -> not (Float.is_finite x)) then
        invalid_arg "Xi components must be finite";
      Array.iteri (fun i xi ->
        let lower, upper = params.bounds.(i) in
        if xi < lower || xi > upper then
          invalid_arg (Printf.sprintf 
            "Xi[%d] = %f outside bounds [%f, %f]" 
            i xi lower upper)
      ) params.values;
      params

    let create_convex_space ~xi_values =
      let bounds = Array.make 6 (0.0, Float.infinity) in
      { values = xi_values; bounds }

    let verify_convexity params =
      let matrix = Array.make_matrix 6 6 0.0 in
      true  
  end

  module Functions = struct
    let compute_a ~xi ~state =
      xi.ParameterSpace.values.(0) +.
      xi.ParameterSpace.values.(1) *. state.StateSpace.a +.
      xi.ParameterSpace.values.(2) *. state.StateSpace.b

    let compute_b ~xi ~state =
      xi.ParameterSpace.values.(3) +.
      xi.ParameterSpace.values.(4) *. state.StateSpace.a +.
      xi.ParameterSpace.values.(5) *. state.StateSpace.b

    let verify_measurable_functions ~xi ~state_seq =
      Array.for_all (fun state ->
        let a_next = compute_a ~xi ~state in
        let b_next = compute_b ~xi ~state in
        a_next > 0.0 && b_next > 0.0
      ) state_seq
  end

  module WellDefinedness = struct
    type validation_result = {
      is_valid: bool;
      error_msg: string option;
    }

    let validate_components params state =
      let is_valid = 
        params.psi > 0.0 && 
        state.StateSpace.a > 0.0 && 
        state.StateSpace.b > 0.0 in
      {
        is_valid;
        error_msg = if is_valid then None 
                   else Some "Invalid component parameters"
      }

    let validate_updates params observations =
      try
        let state = ref (StateSpace.init ~a:params.a_init ~b:params.a_init) in
        Array.iter (fun obs ->
          state := StateSpace.update !state obs
        ) observations;
        { is_valid = true; error_msg = None }
      with _ ->
        { is_valid = false; error_msg = Some "Update validation failed" }

    let check_well_defined params observations =
      let init_state = StateSpace.init ~a:params.a_init ~b:params.a_init in
      let comp_valid = validate_components params init_state in
      if not comp_valid.is_valid then comp_valid
      else validate_updates params observations
  end

  let create ~psi ~a_init ~xi ~require_thinning =
    let xi_params = ParameterSpace.validate_xi xi in
    if require_thinning then
      (* Apply thinning constraints to xi parameters *)
      { psi; a_init; xi = xi_params.values; convex_space = true }
    else
      { psi; a_init; xi = xi_params.values; convex_space = false }

  let update params state obs =
    let xi_params = ParameterSpace.{ 
      values = params.xi; 
      bounds = Array.make 6 (0.0, Float.infinity) 
    } in
    StateSpace.{
      a = Functions.compute_a ~xi:xi_params ~state;
      b = Functions.compute_b ~xi:xi_params ~state
    }
end

module AdaptiveSSM = struct
  type model_params = {
    psi: float;
    a_init: float;
    p_seq: float array;
    q_seq: float array;
  }

  module ParameterValidation = struct
    type validation_result = {
      valid: bool;
      error_msg: string option;
    }

    let validate_p_sequence p_seq =
      let valid = Array.for_all (fun p -> p >= 0.0 && p <= 1.0) p_seq in
      { valid; error_msg = if valid then None else Some "Invalid p sequence" }

    let validate_q_sequence q_seq =
      let valid = Array.for_all (fun q -> q > 0.0) q_seq in
      { valid; error_msg = if valid then None else Some "Invalid q sequence" }

    let check_parameter_compatibility p_seq q_seq =
      if Array.length p_seq <> Array.length q_seq then
        { valid = false; error_msg = Some "Sequence lengths must match" }
      else
        let valid = Array.fold_left2 (fun acc p q ->
          acc && (p +. q > 0.0)
        ) true p_seq q_seq in
        { valid; error_msg = if valid then None else Some "Invalid parameter combination" }
  end

  module StateRecursion = struct
    let update_state params state t obs =
      if t >= Array.length params.p_seq then
        invalid_arg "Time index exceeds parameter sequence length";
      
      let p_t = params.p_seq.(t) in
      let q_t = params.q_seq.(t) in
      
      let intermediate = StateSpace.update state obs in
      StateSpace.{
        a = (p_t +. q_t) *. intermediate.a;
        b = p_t *. intermediate.a +. q_t *. intermediate.b
      }

    let check_stability params state_seq =
      let n = Array.length state_seq in
      if n < 2 then true
      else
        let is_stable = ref true in
        for i = 0 to n - 2 do
          let ratio_a = state_seq.(i+1).StateSpace.a /. state_seq.(i).StateSpace.a in
          let ratio_b = state_seq.(i+1).StateSpace.b /. state_seq.(i).StateSpace.b in
          if abs_float ratio_a > 2.0 || abs_float ratio_b > 2.0 then
            is_stable := false
        done;
        !is_stable
  end

  module Inference = struct
    let forward_filter params observations =
      let n = Array.length observations in
      let states = Array.make n (StateSpace.init ~a:params.a_init ~b:params.a_init) in
      let log_liks = Array.make n 0.0 in
      
      for t = 0 to n - 1 do
        if t > 0 then
          states.(t) <- StateRecursion.update_state params states.(t-1) t observations.(t);
          
        let shape = observations.(t).v /. params.psi in
        let scale = states.(t).b /. 
          (states.(t).a *. observations.(t).mu *. params.psi) in
        
        if shape > 0.0 then
          log_liks.(t) <- Distributions.PearsonVI.(
            log_pdf 
              ~params:{a = shape; b = states.(t).a +. 1.0; scale}
              ~x:observations.(t).y
          );
      done;
      states, log_liks

    let backward_smooth params states observations =
      let n = Array.length states in
      let smoothed = Array.copy states in
      
      for t = n - 2 downto 0 do
        let p_t = params.p_seq.(t) in
        let q_t = params.q_seq.(t) in
        let forward_ratio = smoothed.(t+1).a /. states.(t+1).a in
        
        smoothed.(t) <- StateSpace.{
          a = states.(t).a *. forward_ratio;
          b = states.(t).b *. (p_t *. forward_ratio +. q_t)
        }
      done;
      smoothed
  end

  module StateAnalysis = struct
    type state_dynamics = {
      mean_evolution: float array;
      variance_evolution: float array;
      stability_metric: float;
    }

    let analyze_dynamics states =
      let n = Array.length states in
      let means = Array.map StateSpace.mean states in
      let variances = Array.map StateSpace.variance states in
      
      (* Compute stability metric *)
      let stability = ref 0.0 in
      for t = 0 to n - 2 do
        let mean_change = abs_float (means.(t+1) -. means.(t)) in
        let var_change = abs_float (variances.(t+1) -. variances.(t)) in
        stability := !stability +. mean_change +. var_change;
      done;
      
      {
        mean_evolution = means;
        variance_evolution = variances;
        stability_metric = !stability /. float_of_int (n-1)
      }
  end

  module ModelComparison = struct
    type model_fit = {
      aic: float;
      bic: float;
      dic: float;
    }

    let compute_information_criteria params observations log_liks =
      let n = float_of_int (Array.length observations) in
      let k = 4.0 in  (* Number of parameters: psi, a_init, p, q *)
      
      let total_log_lik = Array.fold_left (+.) 0.0 log_liks in
      
      let aic = -2.0 *. total_log_lik +. 2.0 *. k in
      let bic = -2.0 *. total_log_lik +. k *. log n in
      
      let states, _ = Inference.forward_filter params observations in
      let posterior_means = StateAnalysis.(
        analyze_dynamics states
      ).mean_evolution in
      
      let d_bar = -2.0 *. total_log_lik in
      let d_theta = -2.0 *. Array.fold_left2 (fun acc mean obs ->
        acc +. log (Distributions.PearsonVI.pdf 
          ~params:{a = obs.v /. params.psi;
                   b = params.a_init +. 1.0;
                   scale = mean}
          ~x:obs.y)
      ) 0.0 posterior_means observations in
      
      let p_d = d_bar -. d_theta in
      let dic = d_bar +. 2.0 *. p_d in
      
      { aic; bic; dic }
  end

  module Diagnostics = struct
    type residual_analysis = {
      pearson_residuals: float array;
      deviance_residuals: float array;
      acf: float array;
      qq_points: (float * float) array;
    }

    let analyze_residuals params observations states =
      let n = Array.length observations in
      
      (* Compute Pearson residuals *)
      let pearson = Array.init n (fun i ->
        let expected = observations.(i).mu *. 
          (states.(i).StateSpace.b /. states.(i).StateSpace.a) in
        let std = sqrt (expected *. expected /. states.(i).StateSpace.a) in
        (observations.(i).y -. expected) /. std
      ) in
      
      (* Compute deviance residuals *)
      let deviance = Array.init n (fun i ->
        let y = observations.(i).y in
        let mu = observations.(i).mu *. 
          (states.(i).StateSpace.b /. states.(i).StateSpace.a) in
        if y = 0.0 then 0.0
        else
          let sign = if y > mu then 1.0 else -1.0 in
          sign *. sqrt (2.0 *. ((y *. log (y /. mu)) -. (y -. mu)))
      ) in
      
      (* Compute ACF *)
      let max_lag = min 20 (n / 4) in
      let acf = TimeSeriesAnalysis.compute_acf 
        pearson max_lag in
      
      (* Compute Q-Q plot points *)
      let sorted_residuals = Array.copy pearson in
      Array.sort compare sorted_residuals;
      let qq = Array.init n (fun i ->
        let p = (float_of_int (i + 1)) /. (float_of_int (n + 1)) in
        let theoretical = NumericalUtils.Statistics.normal_quantile p in
        (theoretical, sorted_residuals.(i))
      ) in
      
      { pearson_residuals = pearson;
        deviance_residuals = deviance;
        acf;
        qq_points = qq }

    let validate_model params observations =
      let states, _ = Inference.forward_filter params observations in
      let residuals = analyze_residuals params observations states in
      
      (* Check residual patterns *)
      let has_patterns = ref false in
      
      (* Check autocorrelation *)
      for i = 1 to Array.length residuals.acf - 1 do
        if abs_float residuals.acf.(i) > 2.0 /. 
           sqrt (float_of_int (Array.length observations)) then
          has_patterns := true
      done;
      
      (* Check normality of residuals *)
      let is_normal = DistributionTests.ks_test
        residuals.pearson_residuals
        (fun x -> NumericalUtils.Statistics.normal_cdf x 0.0 1.0) in
      
      (not !has_patterns && is_normal), residuals
  end

  let create ~psi ~a_init ~p_seq ~q_seq =
    let validate_p = ParameterValidation.validate_p_sequence p_seq in
    if not validate_p.valid then
      invalid_arg (Option.get validate_p.error_msg);
    
    let validate_q = ParameterValidation.validate_q_sequence q_seq in
    if not validate_q.valid then
      invalid_arg (Option.get validate_q.error_msg);
    
    let validate_compat = 
      ParameterValidation.check_parameter_compatibility p_seq q_seq in
    if not validate_compat.valid then
      invalid_arg (Option.get validate_compat.error_msg);
    
    if psi <= 0.0 then
      invalid_arg "psi must be positive";
    if a_init <= 0.0 then
      invalid_arg "a_init must be positive";
    
    { psi; a_init; p_seq; q_seq }
end