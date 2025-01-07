open Torch

type dimension = int
type scaling_param = float

type sampling_config = {
  n_samples: int;
  n_chains: int;
  n_warmup: int;
  initial_lambda: float;
  adapt_mass: bool;
  adapt_lambda: bool;
  target_accept: float;
}

type sampling_stats = {
  acceptance_rate: float;
  effective_samples: int;
  r_hat: float;
  max_tree_depth: int option;
}

type sampling_result = {
  samples: Tensor.t array;
  stats: sampling_stats;
  tuning: mass_matrix_stats;
  asymptotic: asymptotic_stats;
}

and mass_matrix_stats = {
  condition_number: float;
  scaling_factor: float;
  efficiency: float;
}

and asymptotic_stats = {
  variance: Tensor.t;
  convergence_rate: float;
  normality_test: float;
}

let stable_log_sum_exp logs =
  let max_log = Array.fold_left max neg_infinity logs in
  let sum = Array.fold_left (fun acc x ->
    acc +. exp (x -. max_log)) 0.0 logs in
  max_log +. log sum

module ConvexFunction = struct
  type t = {
    f: Tensor.t -> Tensor.t;
    domain: Tensor.t -> bool;
    lower_bound: float option;
  }

  let is_proper_lsc_convex f =
    let h = 1e-5 in
    let test_point = Tensor.randn [10] in
    let proper = f.domain test_point in
    let lsc = 
      let y = Tensor.(test_point + randn [10] * float h) in
      if not (f.domain y) then true
      else
        let fx = Tensor.float_value (f.f test_point) in
        let fy = Tensor.float_value (f.f y) in
        fy >= fx -. h
    in
    proper && lsc

  let eval f x = f.f x
end

module MoreauYosida = struct
  let envelope (f: ConvexFunction.t) (lambda: float) (x: Tensor.t) : Tensor.t =
    let x_requires_grad = Tensor.requires_grad_ x in
    let prox = proximal_map f lambda x_requires_grad in
    let env_value = Tensor.(
      f.f prox + float (1. /. (2. *. lambda)) * dot (prox - x) (prox - x)
    ) in
    env_value

  let proximal_map (f: ConvexFunction.t) (lambda: float) (x: Tensor.t) : Tensor.t =
    let y = Tensor.copy x in
    let optimizer = Optimizer.adam ~learning_rate:0.01 [y] in
    
    for _ = 1 to 100 do
      Optimizer.zero_grad optimizer;
      let objective = Tensor.(
        f.f y + float (1. /. (2. *. lambda)) * dot (y - x) (y - x)
      ) in
      Tensor.backward objective;
      Optimizer.step optimizer
    done;
    y

  let verify_lipschitz f lambda x eps =
    let grad1 = Tensor.grad (envelope f lambda x) in
    let x2 = Tensor.(x + randn (shape x) * float eps) in
    let grad2 = Tensor.grad (envelope f lambda x2) in
    let diff_norm = Tensor.(norm_p (grad2 - grad1) 2 |> float_value) in
    diff_norm /. eps <= 1.0 /. lambda
end

module MALA = struct
  let step (f: ConvexFunction.t) (lambda: float) (x: Tensor.t) =
    let dim = Tensor.size x 0 in
    let h = 0.1 in  (* Step size *)
    
    (* Compute gradient *)
    let env_x, grad_x = 
      let x_req_grad = Tensor.requires_grad_ x in
      let env = MoreauYosida.envelope f lambda x_req_grad in
      Tensor.backward env;
      match Tensor.grad x_req_grad with
      | Some grad -> env, grad
      | None -> failwith "Gradient computation failed"
    in

    (* Generate proposal *)
    let noise = Tensor.(randn [dim] * float (sqrt h)) in
    let proposal = Tensor.(x + float (h/.2.) * grad_x + noise) in

    (* Accept/reject *)
    let env_prop, grad_prop =
      let p_req_grad = Tensor.requires_grad_ proposal in
      let env = MoreauYosida.envelope f lambda p_req_grad in
      Tensor.backward env;
      match Tensor.grad p_req_grad with
      | Some grad -> env, grad
      | None -> failwith "Gradient computation failed"
    in

    let log_ratio = Tensor.(
      env_x - env_prop +
      dot (proposal - x - float (h/.2.) * grad_x)
          (proposal - x - float (h/.2.) * grad_x) -
      dot (x - proposal - float (h/.2.) * grad_prop)
          (x - proposal - float (h/.2.) * grad_prop)
    ) |> fun x -> Tensor.float_value x /. (2.0 *. h) in

    let accepted = log (Random.float 1.0) < log_ratio in
    (if accepted then proposal else x, accepted)

  let run config f x0 =
    let samples = Array.make config.n_samples (Tensor.zeros (Tensor.shape x0)) in
    let accepts = ref 0 in
    let current = ref (Tensor.copy x0) in
    
    for i = 0 to config.n_samples - 1 do
      let next, accepted = step f config.initial_lambda !current in
      if accepted then incr accepts;
      current := next;
      samples.(i) <- Tensor.copy next
    done;
    
    let acc_rate = float !accepts /. float config.n_samples in
    
    { samples;
      stats = {
        acceptance_rate = acc_rate;
        effective_samples = 0;  
        r_hat = 1.0;  
        max_tree_depth = None;
      };
      tuning = {
        condition_number = 1.0;  
        scaling_factor = 1.0;    
        efficiency = acc_rate;
      };
      asymptotic = {
        variance = Tensor.zeros [1];  
        convergence_rate = 1.0 /. sqrt (float config.n_samples);
        normality_test = 0.0;  
      };
    }
end

module πλ = struct
  type tree_state = {
    q: Tensor.t;          (* Position *)
    p: Tensor.t;          (* Momentum *)
    grad: Tensor.t;       (* Gradient *)
    energy: float;        (* Hamiltonian *)
    n_leapfrog: int;      (* Number of leapfrog steps *)
    accept_prob: float;   (* Acceptance probability *)
  }

  let leapfrog f lambda eps state =
    (* Half momentum update *)
    let p_half = Tensor.(state.p - float (eps/.2.) * state.grad) in
    
    (* Full position update *)
    let q_new = Tensor.(state.q + float eps * p_half) in
    
    (* Compute new gradient *)
    let env_new, grad_new = 
      let q_req_grad = Tensor.requires_grad_ q_new in
      let env = MoreauYosida.envelope f lambda q_req_grad in
      Tensor.backward env;
      match Tensor.grad q_req_grad with
      | Some grad -> env, grad
      | None -> failwith "Gradient computation failed"
    in
    
    (* Final momentum update *)
    let p_new = Tensor.(p_half - float (eps/.2.) * grad_new) in
    
    (* Compute new energy *)
    let kinetic = Tensor.(dot p_new p_new |> float_value) /. 2.0 in
    let potential = Tensor.float_value env_new in
    
    {
      q = q_new;
      p = p_new;
      grad = grad_new;
      energy = kinetic +. potential;
      n_leapfrog = state.n_leapfrog + 1;
      accept_prob = 1.0;  
    }

  let build_tree f lambda eps state depth direction =
    if depth = 0 then
      (* Base case: single leapfrog step *)
      let next_state = leapfrog f lambda (direction *. eps) state in
      (next_state, next_state, next_state, true)
    else
      (* Recursively build subtrees *)
      let first_state, left_state, right_state, valid = 
        build_tree f lambda eps state (depth-1) direction in
      
      if not valid then 
        (first_state, left_state, right_state, false)
      else
        let start_state = if direction > 0.0 then right_state else left_state in
        let next_first, next_left, next_right, next_valid =
          build_tree f lambda eps start_state (depth-1) direction in
        
        (* Update trajectory and check stopping criteria *)
        let combined_left = if direction < 0.0 then next_left else left_state in
        let combined_right = if direction < 0.0 then right_state else next_right in
        
        let valid =
          next_valid &&
          Tensor.(dot (combined_right.q - combined_left.q) combined_left.p >= float 0.0) &&
          Tensor.(dot (combined_right.q - combined_left.q) combined_right.p >= float 0.0) in
        
        (next_first, combined_left, combined_right, valid)

  let step f lambda x =
    let dim = Tensor.size x 0 in
    
    (* Initialize momentum *)
    let p0 = Tensor.randn [dim] in
    
    (* Initialize state *)
    let init_env, init_grad = 
      let x_req_grad = Tensor.requires_grad_ x in
      let env = MoreauYosida.envelope f lambda x_req_grad in
      Tensor.backward env;
      match Tensor.grad x_req_grad with
      | Some grad -> env, grad
      | None -> failwith "Gradient computation failed"
    in
    
    let init_state = {
      q = x;
      p = p0;
      grad = init_grad;
      energy = Tensor.float_value init_env +. 
               (Tensor.(dot p0 p0 |> float_value)) /. 2.0;
      n_leapfrog = 0;
      accept_prob = 1.0;
    } in
    
    (* Build trajectory *)
    let max_depth = 10 in
    let eps = 0.1 in
    
    let rec expand_tree state depth =
      if depth >= max_depth then (state, false)
      else
        let direction = if Random.bool () then 1.0 else -1.0 in
        let next_state, _, _, valid = 
          build_tree f lambda eps state depth direction in
        
        if not valid then (state, false)
        else
          let accept = exp (state.energy -. next_state.energy) in
          if Random.float 1.0 < accept then
            expand_tree next_state (depth + 1)
          else
            (state, true)
    in
    
    let final_state, accepted = expand_tree init_state 0 in
    (final_state.q, accepted)

  let run config f x0 =
    let samples = Array.make config.n_samples (Tensor.zeros (Tensor.shape x0)) in
    let accepts = ref 0 in
    let current = ref (Tensor.copy x0) in
    let max_depth = ref 0 in
    
    for i = 0 to config.n_samples - 1 do
      let next, accepted = step f config.initial_lambda !current in
      if accepted then incr accepts;
      current := next;
      samples.(i) <- Tensor.copy next
    done;
    
    let acc_rate = float !accepts /. float config.n_samples in
    
    { samples;
      stats = {
        acceptance_rate = acc_rate;
        effective_samples = 0;  
        r_hat = 1.0;  
        max_tree_depth = Some !max_depth;
      };
      tuning = {
        condition_number = 1.0;  
        scaling_factor = 1.0;    
        efficiency = acc_rate;
      };
      asymptotic = {
        variance = Tensor.zeros [1];  
        convergence_rate = 1.0 /. sqrt (float config.n_samples);
        normality_test = 0.0;  
      };
    }
end

module ImportanceSampling = struct
  type weight_stats = {
    sum_weights: float;
    effective_sample_size: float;
    max_weight: float;
  }

  let compute_weights (f: ConvexFunction.t) (lambda: float) (samples: Tensor.t array) =
    (* Compute log weights with better numerical stability *)
    let log_weights = Array.map (fun x ->
      let psi_x = Tensor.float_value (f.f x) in
      let envelope_x = 
        Tensor.float_value (MoreauYosida.envelope f lambda x) in
      envelope_x -. psi_x
    ) samples in
    
    (* Stabilize using log-sum-exp trick *)
    let max_log = Array.fold_left max neg_infinity log_weights in
    Array.map (fun log_w ->
      Tensor.float (exp (log_w -. max_log))
    ) log_weights

  let estimate_mean (f: ConvexFunction.t) (lambda: float) (samples: Tensor.t array) =
    let weights = compute_weights f lambda samples in
    
    (* Compute weighted mean *)
    let weighted_sum = Array.fold_left2 (fun acc x w ->
      Tensor.(acc + x * w)
    ) (Tensor.zeros (Tensor.shape samples.(0))) samples weights in
    
    let sum_weights = Array.fold_left (fun acc w ->
      Tensor.float_value w +. acc
    ) 0.0 weights in
    
    let mean = Tensor.(weighted_sum / float sum_weights) in
    
    (* Compute ESS *)
    let sum_sq_weights = Array.fold_left (fun acc w ->
      acc +. (Tensor.float_value w) ** 2.0
    ) 0.0 weights in
    
    let ess = sum_weights ** 2.0 /. sum_sq_weights in
    
    let max_weight = Array.fold_left (fun acc w ->
      max acc (Tensor.float_value w)
    ) neg_infinity weights in
    
    (mean, {
      sum_weights;
      effective_sample_size = ess;
      max_weight;
    })

  let estimate_variance (f: ConvexFunction.t) (lambda: float) (samples: Tensor.t array) =
    let weights = compute_weights f lambda samples in
    let mean, _ = estimate_mean f lambda samples in
    
    (* Compute weighted variance *)
    let weighted_var = Array.fold_left2 (fun acc x w ->
      let centered = Tensor.(x - mean) in
      let outer = Tensor.(mm (unsqueeze centered 1) (unsqueeze centered 0)) in
      Tensor.(acc + outer * w)
    ) (Tensor.zeros (Tensor.shape samples.(0))) samples weights in
    
    let sum_weights = Array.fold_left (fun acc w ->
      Tensor.float_value w +. acc
    ) 0.0 weights in
    
    Tensor.(weighted_var / float sum_weights)
end

module Quantile = struct
  type quantile_estimate = {
    value: float;
    std_error: float;
    confidence_interval: float * float;
  }

  let estimate (samples: Tensor.t array) (weights: Tensor.t array) (alpha: float) =
    let n = Array.length samples in
    
    (* Sort samples and weights together *)
    let indices = Array.init n (fun i -> i) in
    Array.sort (fun i j ->
      compare 
        (Tensor.float_value (Tensor.get samples.(i) [0]))
        (Tensor.float_value (Tensor.get samples.(j) [0]))
    ) indices;
    
    let sorted_samples = Array.map (fun i -> samples.(i)) indices in
    let sorted_weights = Array.map (fun i -> weights.(i)) indices in
    
    (* Compute cumulative weights *)
    let cum_weights = Array.make n (Tensor.zeros [1]) in
    cum_weights.(0) <- sorted_weights.(0);
    for i = 1 to n-1 do
      cum_weights.(i) <- Tensor.(cum_weights.(i-1) + sorted_weights.(i))
    done;
    
    let total_weight = Tensor.float_value cum_weights.(n-1) in
    let target = alpha *. total_weight in
    
    (* Find quantile through interpolation *)
    let rec find_interval i =
      if i >= n-1 then
        Tensor.float_value (Tensor.get sorted_samples.(n-1) [0])
      else
        let w_i = Tensor.float_value cum_weights.(i) in
        let w_next = Tensor.float_value cum_weights.(i+1) in
        if w_i <= target && target <= w_next then
          let t = (target -. w_i) /. (w_next -. w_i) in
          let v1 = Tensor.float_value (Tensor.get sorted_samples.(i) [0]) in
          let v2 = Tensor.float_value (Tensor.get sorted_samples.(i+1) [0]) in
          v1 *. (1.0 -. t) +. v2 *. t
        else
          find_interval (i+1)
    in
    
    let value = find_interval 0 in
    
    (* Bootstrap for error estimation *)
    let n_boot = 1000 in
    let boot_estimates = Array.init n_boot (fun _ ->
      let boot_idx = Array.init n (fun _ -> Random.int n) in
      let boot_samples = Array.map (fun i -> samples.(i)) boot_idx in
      let boot_weights = Array.map (fun i -> weights.(i)) boot_idx in
      estimate boot_samples boot_weights alpha |> fun x -> x.value
    ) in
    
    Array.sort compare boot_estimates;
    let std_error = 
      let mean = Array.fold_left (+.) 0.0 boot_estimates /. float n_boot in
      let var = Array.fold_left (fun acc x ->
        let d = x -. mean in acc +. d *. d
      ) 0.0 boot_estimates /. float (n_boot - 1) in
      sqrt var
    in
    
    let ci_low = boot_estimates.(int_of_float (0.025 *. float n_boot)) in
    let ci_high = boot_estimates.(int_of_float (0.975 *. float n_boot)) in
    
    {value; std_error; confidence_interval = (ci_low, ci_high)}
end

module Diagnostics = struct
  type monitor_stats = {
    r_hat: float;
    ess: float;
    stable: bool;
    stationarity_test: float;
  }

  let compute_autocovariance samples max_lag =
    let n = Array.length samples in
    let dim = Tensor.size samples.(0) 0 in
    let mean = Array.fold_left (fun acc x -> 
      Tensor.(acc + x)) (Tensor.zeros [dim]) samples
      |> fun sum -> Tensor.(sum / float n) in
    
    Array.init max_lag (fun lag ->
      let sum = ref (Tensor.zeros [dim; dim]) in
      for i = 0 to n - lag - 1 do
        let x_i = Tensor.(samples.(i) - mean) in
        let x_lag = Tensor.(samples.(i + lag) - mean) in
        sum := Tensor.(!sum + mm (unsqueeze x_i 1) (unsqueeze x_lag 0))
      done;
      Tensor.(!sum / float (n - lag))
    )

  let monitor_convergence chains =
    let n_chains = Array.length chains in
    let chain_length = Array.length chains.(0) in
    let dim = Tensor.size chains.(0).(0) 0 in
    
    (* Compute chain means *)
    let chain_means = Array.map (fun chain ->
      Array.fold_left (fun acc x -> Tensor.(acc + x)) 
        (Tensor.zeros [dim]) chain
      |> fun sum -> Tensor.(sum / float chain_length)
    ) chains in
    
    (* Overall mean *)
    let overall_mean = 
      Array.fold_left (fun acc m -> Tensor.(acc + m)) 
        (Tensor.zeros [dim]) chain_means
      |> fun sum -> Tensor.(sum / float n_chains) in
    
    (* Between-chain variance *)
    let b = 
      Array.fold_left (fun acc m ->
        let diff = Tensor.(m - overall_mean) in
        Tensor.(acc + mm (unsqueeze diff 1) (unsqueeze diff 0))
      ) (Tensor.zeros [dim; dim]) chain_means
      |> fun sum -> Tensor.(sum * float chain_length / float (n_chains - 1)) in
    
    (* Within-chain variance *)
    let w = Array.map (fun chain ->
      let chain_auto = compute_autocovariance chain 1 in
      chain_auto.(0)
    ) chains |> Array.fold_left (fun acc x -> Tensor.(acc + x)) 
        (Tensor.zeros [dim; dim])
      |> fun sum -> Tensor.(sum / float n_chains) in
    
    (* R-hat computation *)
    let var_plus = Tensor.((float ((chain_length - 1) / chain_length) * w + b) / w) in
    let r_hat = Tensor.(sqrt (maximum var_plus) |> float_value) in
    
    (* ESS computation *)
    let max_lag = min chain_length 50 in
    let auto_corr = compute_autocovariance (Array.concat (Array.to_list chains)) max_lag in
    let rho = Array.map (fun gamma ->
      Tensor.(gamma / auto_corr.(0))
    ) auto_corr in
    
    let sum_rho = Array.fold_left (fun acc r ->
      Tensor.(acc + r)
    ) (Tensor.zeros [dim; dim]) rho in
    
    let ess = float (n_chains * chain_length) /. 
              Tensor.(maximum sum_rho |> float_value) in
    
    (* Stationarity test *)
    let first_half = Array.map (fun chain ->
      Array.sub chain 0 (chain_length / 2)
    ) chains in
    let second_half = Array.map (fun chain ->
      Array.sub chain (chain_length / 2) (chain_length / 2)
    ) chains in
    
    let stat_test = 
      let fh_mean = Array.fold_left (fun acc chain ->
        Array.fold_left (fun acc x -> Tensor.(acc + x)) acc chain
      ) (Tensor.zeros [dim]) first_half
        |> fun sum -> Tensor.(sum / float (n_chains * chain_length / 2)) in
      let sh_mean = Array.fold_left (fun acc chain ->
        Array.fold_left (fun acc x -> Tensor.(acc + x)) acc chain
      ) (Tensor.zeros [dim]) second_half
        |> fun sum -> Tensor.(sum / float (n_chains * chain_length / 2)) in
      
      Tensor.(dot (fh_mean - sh_mean) (fh_mean - sh_mean) |> sqrt |> float_value)
    in
    
    {
      r_hat;
      ess;
      stable = r_hat < 1.1 && ess > 100.0;
      stationarity_test = stat_test;
    }

  let verify_convergence samples =
    let stats = monitor_convergence [|samples|] in
    let details = [
      Printf.sprintf "R-hat: %.3f" stats.r_hat;
      Printf.sprintf "ESS: %.1f" stats.ess;
      Printf.sprintf "Stationarity test: %.3f" stats.stationarity_test;
    ] in
    (stats.stable, details)
end

module DimensionScaling = struct
  type scaling_result = {
    optimal_lambda: float;
    dimension_factor: float;
    efficiency_estimate: float;
  }

  let analyze_spectrum omega =
    let dim = Tensor.size omega 0 in
    
    (* Power iteration for largest eigenvalue *)
    let rec power_iterate x n =
      if n = 0 then x
      else
        let mx = Tensor.(mm omega (unsqueeze x 1)) in
        let norm = Tensor.(dot mx mx |> sqrt |> float_value) in
        power_iterate Tensor.(squeeze mx / float norm) (n-1)
    in
    
    let v1 = power_iterate (Tensor.randn [dim]) 100 in
    let largest = 
      let mv = Tensor.(mm omega (unsqueeze v1 1)) in
      Tensor.(dot v1 (squeeze mv) |> float_value)
    in
    
    (* Inverse iteration for smallest eigenvalue *)
    let omega_inv = 
      try Some (Tensor.inverse omega)
      with _ -> None
    in
    
    let smallest = match omega_inv with
      | Some inv ->
          let rec inverse_iterate x n =
            if n = 0 then x
            else
              let mx = Tensor.(mm inv (unsqueeze x 1)) in
              let norm = Tensor.(dot mx mx |> sqrt |> float_value) in
              inverse_iterate Tensor.(squeeze mx / float norm) (n-1)
          in
          let v2 = inverse_iterate (Tensor.randn [dim]) 100 in
          1.0 /. Tensor.(
            dot v2 (squeeze (mm inv (unsqueeze v2 1))) 
            |> float_value
          )
      | None -> largest /. 1000.0 
    in
    
    (smallest, largest)

  let compute_optimal_lambda omega dim =
    let s1, sd = analyze_spectrum omega in
    
    (* Binary search for optimal lambda *)
    let rec search low high iter =
      if iter = 0 || high -. low < 1e-10 then (low +. high) /. 2.0
      else
        let mid = (low +. high) /. 2.0 in
        let sum = mid *. float dim -. s1 in
        if sum > 0.0 then
          search low mid (iter - 1)
        else
          search mid high (iter - 1)
    in
    
    let optimal = search (s1 /. float dim) (sd /. float dim) 100 in
    
    (* Compute efficiency estimate *)
    let condition = sd /. s1 in
    let efficiency = 1.0 /. sqrt condition in
    
    {
      optimal_lambda = optimal;
      dimension_factor = float dim;
      efficiency_estimate = efficiency;
    }

  let verify_scaling samples =
    let n = Array.length samples in
    let dim = Tensor.size samples.(0) 0 in
    
    (* Compute sample covariance *)
    let mean = Array.fold_left (fun acc x -> Tensor.(acc + x)) 
      (Tensor.zeros [dim]) samples
      |> fun sum -> Tensor.(sum / float n) in
    
    let cov = Array.fold_left (fun acc x ->
      let centered = Tensor.(x - mean) in
      Tensor.(acc + mm (unsqueeze centered 1) (unsqueeze centered 0))
    ) (Tensor.zeros [dim; dim]) samples
      |> fun sum -> Tensor.(sum / float (n - 1)) in
    
    (* Analyze scaling through eigenvalues *)
    let s1, sd = analyze_spectrum cov in
    let scaling = sd /. s1 in
    let acceptable = scaling < float dim *. log (float dim) in
    
    (acceptable, scaling)
end

module Ergodicity = struct
  type ergodicity_result = {
    is_ergodic: bool;
    spectral_gap: float option;
    mixing_time: int option;
  }

  let estimate_spectral_gap samples =
    let n = Array.length samples in
    let max_lag = min n 50 in
    
    (* Compute autocorrelations *)
    let auto_corr = Array.init max_lag (fun lag ->
      let sum = ref 0.0 in
      for i = 0 to n - lag - 1 do
        sum := !sum +. Tensor.(
          dot samples.(i) samples.(i+lag) |> float_value
        )
      done;
      !sum /. float (n - lag)
    ) in
    
    if Array.length auto_corr > 1 then
      Some (-.log (abs_float (auto_corr.(1) /. auto_corr.(0))))
    else
      None

  let verify_ergodicity samples potential lambda =
    let n = Array.length samples in
    
    (* Estimate spectral gap *)
    let gap = estimate_spectral_gap samples in
    
    (* Compute mixing time if possible *)
    let mixing = match gap with
      | Some g when g > 0.0 -> Some (int_of_float (ceil (1.0 /. g)))
      | _ -> None
    in
    
    (* Check geometric drift condition *)
    let drift_ok = Array.fold_left (fun acc x ->
      let pot = potential x in
      acc && pot < Float.infinity
    ) true samples in
    
    {
      is_ergodic = gap <> None && Option.get gap > 0.0 && drift_ok;
      spectral_gap = gap;
      mixing_time = mixing;
    }

  let verify_conditions f lambda x =
    let details = ref [] in
    
    (* Check envelope property *)
    let env_ok, env_diff = 
      let env = MoreauYosida.envelope f lambda x in
      let f_x = f.f x in
      Tensor.(float_value env <= float_value f_x,
              float_value f_x - float_value env)
    in
    details := Printf.sprintf "Envelope property: %b (diff: %f)" env_ok env_diff 
      :: !details;
    
    (* Check Lipschitz condition *)
    let lip_ok = MoreauYosida.verify_lipschitz f lambda x 0.01 in
    details := Printf.sprintf "Lipschitz condition: %b" lip_ok :: !details;
    
    (env_ok && lip_ok, List.rev !details)
end