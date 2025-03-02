open Torch

(* Sample from a normal distribution *)
let normal_sample ~mean ~std ~shape =
  let z = Tensor.randn shape ~dtype:(Tensor.kind mean) ~device:(Tensor.device mean) in
  Tensor.(mean + z * std)

(* Log probability density of normal distribution *)
let normal_logpdf x ~mean ~std =
  let pi = Scalar.f 3.14159265358979323846 in
  let n = Tensor.shape x |> List.fold_left (fun acc x -> acc * x) 1 in
  let log_std = Tensor.log std in
  let term1 = Tensor.((Scalar.f (-0.5) * Scalar.f (float_of_int n)) * Tensor.log (Scalar.f 2. * pi)) in
  let term2 = Tensor.(Scalar.f (-0.5) * Scalar.f (float_of_int n) * Scalar.f 2. * log_std) in
  let term3 = Tensor.(Scalar.f (-0.5) * Tensor.sum (Tensor.pow_scalar (x - mean) (Scalar.f 2.)) / (std * std)) in
  Tensor.(term1 + term2 + term3)

(* Metropolis-Hastings algorithm step *)
let metropolis_step current_sample ~proposal_fn ~log_prob_fn ~step_size =
  let proposed_sample = proposal_fn current_sample step_size in
  let current_log_prob = log_prob_fn current_sample in
  let proposed_log_prob = log_prob_fn proposed_sample in
  let log_acceptance_ratio = Tensor.(proposed_log_prob - current_log_prob) in
  
  let u = Tensor.rand [] ~dtype:(Tensor.kind current_log_prob) in
  let log_u = Tensor.log u in
  
  if Tensor.to_float0_exn Tensor.(log_u < log_acceptance_ratio) then
    (proposed_sample, true)  (* accepted *)
  else
    (current_sample, false)  (* rejected *)

(* Random walk proposal *)
let random_walk_proposal sample step_size =
  let noise = normal_sample 
    ~mean:(Tensor.zeros_like sample) 
    ~std:step_size 
    ~shape:(Tensor.shape sample) in
  Tensor.(sample + noise)

(* Replica Exchange Monte Carlo *)
module REMC = struct
  type params = {
    n_replicas: int;
    inverse_temperatures: Tensor.t;
    step_sizes: Tensor.t;
    burn_in: int;
    n_samples: int;
  }
  
  (* Create a set of inverse temperatures *)
  let create_temperatures ~min_temp ~max_temp ~n_replicas =
    let gamma = (max_temp /. min_temp) ** (1. /. float_of_int (n_replicas - 1)) in
    let temps = Array.make n_replicas 0. in
    for i = 0 to n_replicas - 1 do
      temps.(i) <- min_temp *. (gamma ** float_of_int i)
    done;
    Tensor.of_float1 temps
  
  (* Perform replica exchange between two replicas *)
  let exchange_replicas ~replica_i ~replica_j ~beta_i ~beta_j ~log_prob_fn =
    let log_prob_i_at_i = log_prob_fn replica_i beta_i in
    let log_prob_j_at_j = log_prob_fn replica_j beta_j in
    let log_prob_i_at_j = log_prob_fn replica_i beta_j in
    let log_prob_j_at_i = log_prob_fn replica_j beta_i in
    
    let log_ratio = Tensor.(
      (log_prob_j_at_i + log_prob_i_at_j) - (log_prob_i_at_i + log_prob_j_at_j)
    ) in
    
    let u = Tensor.rand [] ~dtype:(Tensor.kind log_ratio) in
    let log_u = Tensor.log u in
    
    if Tensor.to_float0_exn Tensor.(log_u < log_ratio) then
      (replica_j, replica_i, true)  (* Swapped *)
    else
      (replica_i, replica_j, false)  (* Not swapped *)
      
  (* Calculate Bayesian free energy difference *)
  let calculate_free_energy_diff ~samples ~beta_i ~beta_j ~log_prob_fn =
    let n = List.length samples in
    let energy_diffs = List.map (fun sample ->
      let log_prob_beta_i = log_prob_fn sample beta_i in
      let log_prob_beta_j = log_prob_fn sample beta_j in
      Tensor.(log_prob_beta_j - log_prob_beta_i)
    ) samples in
    
    let sum_exp_energy_diffs = 
      List.fold_left (fun acc diff -> 
        Tensor.(acc + Tensor.exp diff)
      ) (Tensor.zeros []) energy_diffs in
    
    Tensor.(- (log (sum_exp_energy_diffs / Scalar.f (float_of_int n))))
  
  (* REMC algorithm *)
  let run ~initial_samples ~log_prob_fn ~proposal_fn params =
    let n_replicas = params.n_replicas in
    let replicas = Array.copy initial_samples in
    let all_samples = Array.make n_replicas [] in
    
    for t = 1 to params.burn_in + params.n_samples do
      (* Update each replica using Metropolis step *)
      for i = 0 to n_replicas - 1 do
        let beta = Tensor.get params.inverse_temperatures [i] |> Tensor.to_float0_exn in
        let step_size = Tensor.get params.step_sizes [i] |> Tensor.to_float0_exn in
        
        let log_prob = fun sample ->
          log_prob_fn sample (Tensor.of_float0 beta)
        in
        
        let (new_sample, accepted) = 
          MCMC.metropolis_step replicas.(i) 
            ~proposal_fn:(fun sample step_size -> 
              proposal_fn sample (Tensor.of_float0 step_size))
            ~log_prob_fn:log_prob
            ~step_size:(Tensor.of_float0 step_size)
        in
        
        replicas.(i) <- new_sample
      done;
      
      (* Perform exchanges between adjacent replicas *)
      for i = 0 to n_replicas - 2 do
        let j = i + 1 in
        let beta_i = Tensor.get params.inverse_temperatures [i] in
        let beta_j = Tensor.get params.inverse_temperatures [j] in
        
        let (new_replica_i, new_replica_j, _) = 
          exchange_replicas 
            ~replica_i:replicas.(i) 
            ~replica_j:replicas.(j)
            ~beta_i:(Tensor.of_float0 (Tensor.to_float0_exn beta_i))
            ~beta_j:(Tensor.of_float0 (Tensor.to_float0_exn beta_j))
            ~log_prob_fn
        in
        
        replicas.(i) <- new_replica_i;
        replicas.(j) <- new_replica_j
      done;
      
      (* Store samples after burn-in *)
      if t > params.burn_in then
        for i = 0 to n_replicas - 1 do
          all_samples.(i) <- replicas.(i) :: all_samples.(i)
        done
    done;
    
    all_samples
end

(* Sequential Monte Carlo Samplers *)
module SMCS = struct
  type params = {
    n_samples: int;
    n_mcmc_steps: int;
    step_sizes: Tensor.t;
  }
  
  (* Resample particles according to weights *)
  let resample particles weights =
    let n = Array.length particles in
    let cum_weights = Array.make n 0. in
    cum_weights.(0) <- weights.(0);
    for i = 1 to n - 1 do
      cum_weights.(i) <- cum_weights.(i-1) +. weights.(i)
    done;
    
    let total_weight = cum_weights.(n-1) in
    let normalized_cum_weights = Array.map (fun w -> w /. total_weight) cum_weights in
    
    let new_particles = Array.make n (Tensor.zeros []) in
    for i = 0 to n - 1 do
      let u = Random.float 1.0 in
      let j = ref 0 in
      while !j < n - 1 && normalized_cum_weights.(!j) < u do
        incr j
      done;
      new_particles.(i) <- particles.(!j)
    done;
    
    new_particles
    
  (* Calculate ESS (Effective Sample Size) *)
  let calculate_ess weights =
    let n = Array.length weights in
    let sum_weights = Array.fold_left (+.) 0. weights in
    
    let normalized_weights = Array.map (fun w -> w /. sum_weights) weights in
    let sum_squared_weights = 
      Array.fold_left (fun acc w -> acc +. (w *. w)) 0. normalized_weights in
    
    1. /. sum_squared_weights
  
  (* SMC algorithm *)
  let run ~initial_samples ~log_prior_fn ~log_likelihood_fn ~proposal_fn params =
    let n = Array.length initial_samples in
    let particles = Array.copy initial_samples in
    let beta = ref 0. in  (* Start with prior (beta=0) *)
    let all_samples = ref [] in
    
    while !beta < 1.0 do
      (* Determine next inverse temperature *)
      let next_beta = ref (!beta +. 0.1) in
      if !next_beta > 1.0 then next_beta := 1.0;
      
      (* Calculate weights for resampling *)
      let weights = Array.map (fun particle ->
        let log_likelihood = log_likelihood_fn particle in
        Tensor.to_float0_exn (Tensor.exp (Tensor.of_float0 (!next_beta -. !beta) * log_likelihood))
      ) particles in
      
      (* Resample particles *)
      let new_particles = resample particles weights in
      
      (* Update particles with MCMC steps *)
      for i = 0 to n - 1 do
        let mut_particle = ref new_particles.(i) in
        
        for _ = 1 to params.n_mcmc_steps do
          let step_size = 
            Tensor.get params.step_sizes [0] |> Tensor.to_float0_exn in
          
          let log_prob = fun sample ->
            let log_prior = log_prior_fn sample in
            let log_likelihood = log_likelihood_fn sample in
            Tensor.(log_prior + (Tensor.of_float0 !next_beta * log_likelihood))
          in
          
          let (new_sample, _) = 
            MCMC.metropolis_step !mut_particle
              ~proposal_fn:(fun sample step_size -> 
                proposal_fn sample (Tensor.of_float0 step_size))
              ~log_prob_fn:log_prob
              ~step_size:(Tensor.of_float0 step_size)
          in
          
          mut_particle := new_sample
        done;
        
        new_particles.(i) <- !mut_particle
      done;
      
      (* Store samples and update beta *)
      if !next_beta = 1.0 then
        all_samples := Array.to_list new_particles;
      
      beta := !next_beta;
      Array.blit new_particles 0 particles 0 n
    done;
    
    !all_samples
end

(* Waste-free Sequential Monte Carlo *)
module WasteFree_SMC = struct
  type params = {
    n_samples: int;
    n_chains: int;
    n_mcmc_steps: int;
    step_sizes: Tensor.t;
  }
  
  (* Waste-free SMC algorithm *)
  let run ~initial_samples ~log_prior_fn ~log_likelihood_fn ~proposal_fn params =
    let n = params.n_samples in
    let s = params.n_chains in
    let particles = Array.copy initial_samples in
    let beta = ref 0. in
    let all_samples = ref [] in
    
    while !beta < 1.0 do
      (* Determine next inverse temperature *)
      let next_beta = ref (!beta +. 0.1) in
      if !next_beta > 1.0 then next_beta := 1.0;
      
      (* Calculate weights for resampling *)
      let weights = Array.map (fun particle ->
        let log_likelihood = log_likelihood_fn particle in
        Tensor.to_float0_exn (Tensor.exp (Tensor.of_float0 (!next_beta -. !beta) * log_likelihood))
      ) particles in
      
      (* Resample S particles *)
      let selected_indices = Array.make s 0 in
      let cum_weights = Array.make n 0. in
      cum_weights.(0) <- weights.(0);
      for i = 1 to n - 1 do
        cum_weights.(i) <- cum_weights.(i-1) +. weights.(i)
      done;
      
      let total_weight = cum_weights.(n-1) in
      let normalized_cum_weights = Array.map (fun w -> w /. total_weight) cum_weights in
      
      for i = 0 to s - 1 do
        let u = Random.float 1.0 in
        let j = ref 0 in
        while !j < n - 1 && normalized_cum_weights.(!j) < u do
          incr j
        done;
        selected_indices.(i) <- !j
      done;
      
      (* Create new particles array *)
      let new_particles = Array.make n (Tensor.zeros []) in
      
      (* Perform MCMC steps for each chain *)
      for chain_idx = 0 to s - 1 do
        let source_idx = selected_indices.(chain_idx) in
        let mut_particle = ref particles.(source_idx) in
        
        for step = 0 to params.n_mcmc_steps - 1 do
          let step_size = 
            Tensor.get params.step_sizes [0] |> Tensor.to_float0_exn in
          
          let log_prob = fun sample ->
            let log_prior = log_prior_fn sample in
            let log_likelihood = log_likelihood_fn sample in
            Tensor.(log_prior + (Tensor.of_float0 !next_beta * log_likelihood))
          in
          
          let (new_sample, _) = 
            MCMC.metropolis_step !mut_particle
              ~proposal_fn:(fun sample step_size -> 
                proposal_fn sample (Tensor.of_float0 step_size))
              ~log_prob_fn:log_prob
              ~step_size:(Tensor.of_float0 step_size)
          in
          
          mut_particle := new_sample;
          
          (* Store intermediate samples *)
          if step < params.n_mcmc_steps - 1 then
            new_particles.(chain_idx * params.n_mcmc_steps + step) <- Tensor.copy !mut_particle
        done;
        
        (* Store final sample *)
        new_particles.(chain_idx * params.n_mcmc_steps + params.n_mcmc_steps - 1) <- !mut_particle
      done;
      
      (* Store samples and update beta *)
      if !next_beta = 1.0 then
        all_samples := Array.to_list new_particles;
      
      beta := !next_beta;
      Array.blit new_particles 0 particles 0 n
    done;
    
    !all_samples
end

(* Robbins-Monro algorithm for step size adaptation *)
module RobbinsMonro = struct
  type params = {
    c: float;
    n0: int;
    target_acceptance: float;
  }
  
  (* Update step size based on acceptance rate *)
  let update_step_size ~current_step_size ~acceptance_rate ~params ~iteration =
    let c = params.c in
    let n0 = float_of_int params.n0 in
    let p_target = params.target_acceptance in
    
    current_step_size *. (1.0 +. c *. (acceptance_rate -. p_target) /. (n0 +. float_of_int iteration))
end

(* Calculate entropy of a distribution *)
let calculate_entropy samples ~log_prob_fn =
  let n = List.length samples in
  let sum_log_prob = List.fold_left (fun acc sample ->
    let log_prob = log_prob_fn sample |> Tensor.to_float0_exn in
    acc +. log_prob
  ) 0.0 samples in
  
  (-1.0) *. sum_log_prob /. float_of_int n

(* Calculate free energy difference between two temperature states *)
let calculate_free_energy_diff ~samples ~beta_i ~beta_j ~log_likelihood_fn =
  let n = List.length samples in
  let sum_exp_energy_diff = List.fold_left (fun acc sample ->
    let log_likelihood = log_likelihood_fn sample |> Tensor.to_float0_exn in
    let energy_diff = (beta_j -. beta_i) *. log_likelihood in
    acc +. exp energy_diff
  ) 0.0 samples in
  
  (-1.0) *. log (sum_exp_energy_diff /. float_of_int n)

(* Calculate free energy using thermodynamic integration *)
let thermodynamic_integration ~samples_at_betas ~betas ~log_likelihood_fn =
  let n_temps = Array.length betas in
  if n_temps < 2 then failwith "Need at least 2 temperatures for integration";
  
  let free_energy = ref 0.0 in
  
  for i = 0 to n_temps - 2 do
    let beta_i = betas.(i) in
    let beta_j = betas.(i+1) in
    
    let samples_i = samples_at_betas.(i) in
    
    (* Calculate ⟨E⟩ at temperature beta_i *)
    let avg_energy_i = List.fold_left (fun acc sample ->
      let log_likelihood = log_likelihood_fn sample |> Tensor.to_float0_exn in
      acc -. log_likelihood  (* Energy = -log(likelihood) *)
    ) 0.0 samples_i /. float_of_int (List.length samples_i) in
    
    (* Trapezoid rule for integration *)
    let delta_beta = beta_j -. beta_i in
    free_energy := !free_energy +. avg_energy_i *. delta_beta
  done;
  
  !free_energy

(* Sequential Exchange Monte Carlo *)
module SEMC = struct
  (* Parameters *)
  type params = {
    n_samples: int;                (* Number of samples *)
    n_parallel: int;               (* Number of parallel chains *)
    target_exchange_rate: float;   (* Target exchange rate between temperatures *)
    step_size_params: {
      initial: float;              (* Initial step size *)
      adaptation_constant: float;  (* Robbins-Monro adaptation constant *)
      adaptation_offset: int;      (* Robbins-Monro adaptation offset *)
      target_acceptance: float;    (* Target acceptance rate for Metropolis steps *)
    };
  }
  
  (* Default parameters *)
  let default_params = {
    n_samples = 1000;
    n_parallel = 50;
    target_exchange_rate = 0.3;
    step_size_params = {
      initial = 1.0;
      adaptation_constant = 4.0;
      adaptation_offset = 15;
      target_acceptance = 0.5;
    };
  }
  
  (* Determine next inverse temperature to maintain constant exchange rate *)
  let determine_next_beta ~current_beta ~samples ~log_likelihood_fn ~target_exchange_rate =
    if current_beta >= 1.0 then 1.0
    else
      (* Function to calculate exchange rate for a proposed beta difference *)
      let calculate_exchange_rate delta_beta =
        let next_beta = min 1.0 (current_beta +. delta_beta) in
        if delta_beta <= 0.0 then 1.0
        else
          (* Calculate log probabilities at current and next beta *)
          let log_probs_current = Array.map (fun sample ->
            let log_likelihood = log_likelihood_fn sample |> Tensor.to_float0_exn in
            current_beta *. log_likelihood
          ) samples in
          
          let log_probs_next = Array.map (fun sample ->
            let log_likelihood = log_likelihood_fn sample |> Tensor.to_float0_exn in
            next_beta *. log_likelihood
          ) samples in
          
          (* Calculate exchange probability across all sample pairs *)
          let n = Array.length samples in
          let sum_exchange_prob = ref 0.0 in
          let count = ref 0 in
          
          for i = 0 to min 100 n - 1 do  (* Limit computation for large sample sizes *)
            for j = 0 to min 100 n - 1 do
              if i <> j then
                let log_ratio = (log_probs_next.(i) -. log_probs_current.(i)) +.
                              (log_probs_current.(j) -. log_probs_next.(j)) in
                sum_exchange_prob := !sum_exchange_prob +. min 1.0 (exp log_ratio);
                incr count
              end
            done
          done;
          
          if !count = 0 then 0.0 else !sum_exchange_prob /. float_of_int !count
      in
      
      (* Binary search to find appropriate beta *)
      let rec binary_search min_diff max_diff =
        (* Exit condition for binary search *)
        if max_diff -. min_diff < 0.001 then
          let beta_diff = (min_diff +. max_diff) /. 2.0 in
          min 1.0 (current_beta +. beta_diff)
        else
          let mid_diff = (min_diff +. max_diff) /. 2.0 in
          let exchange_rate = calculate_exchange_rate mid_diff in
          
          if exchange_rate < target_exchange_rate then
            (* Need smaller temperature difference *)
            binary_search min_diff mid_diff
          else
            (* Can try larger temperature difference *)
            binary_search mid_diff max_diff
      in
      
      (* Start binary search with reasonable range *)
      let max_diff = min 0.5 (1.0 -. current_beta) in
      binary_search 0.001 max_diff
  
  (* Update step size using Robbins-Monro algorithm *)
  let update_step_size ~current_step_size ~acceptance_rate ~params ~iteration =
    let c = params.step_size_params.adaptation_constant in
    let n0 = float_of_int params.step_size_params.adaptation_offset in
    let p_target = params.step_size_params.target_acceptance in
    
    current_step_size *. (1.0 +. c *. (acceptance_rate -. p_target) /. (n0 +. float_of_int iteration))
  
  (* Calculate weights for resampling *)
  let calculate_weights ~samples ~log_likelihood_fn ~beta_prev ~beta_next =
    Array.map (fun sample ->
      let log_likelihood = log_likelihood_fn sample |> Tensor.to_float0_exn in
      exp ((beta_next -. beta_prev) *. log_likelihood)
    ) samples
  
  (* Main SEMC *)
  let run ~initial_samples ~log_prior_fn ~log_likelihood_fn ~proposal_fn ?(params=default_params) ?(verbose=true) () =
    let n = params.n_samples in
    let s = params.n_parallel in
    let particles = Array.copy initial_samples in
    let beta = ref 0.0 in  (* Start with prior (beta=0) *)
    let beta_schedule = ref [0.0] in
    let all_samples = ref [] in
    let step_size = ref params.step_size_params.initial in
    let acceptance_count = ref 0 in
    let total_updates = ref 0 in
    
    if verbose then
      Printf.printf "Starting SEMC with %d samples and %d parallel chains\n%!" n s;
    
    let start_time = Unix.gettimeofday () in
    
    (* Continue until beta reaches 1.0 *)
    while !beta < 1.0 do
      (* Determine next inverse temperature *)
      let next_beta = determine_next_beta 
        ~current_beta:!beta 
        ~samples:particles 
        ~log_likelihood_fn 
        ~target_exchange_rate:params.target_exchange_rate in
      
      beta_schedule := next_beta :: !beta_schedule;
      if verbose then
        Printf.printf "Temperature step: %.4f -> %.4f\n%!" !beta next_beta;
      
      (* Calculate weights for resampling *)
      let weights = calculate_weights 
        ~samples:particles 
        ~log_likelihood_fn 
        ~beta_prev:!beta 
        ~beta_next:next_beta in
      
      (* Normalize weights *)
      let total_weight = Array.fold_left (+.) 0.0 weights in
      let normalized_weights = Array.map (fun w -> w /. total_weight) weights in
      
      (* Calculate ESS (Effective Sample Size) *)
      let sum_squared_weights = Array.fold_left (fun acc w -> acc +. (w *. w)) 0.0 normalized_weights in
      let ess = 1.0 /. sum_squared_weights in
      if verbose then
        Printf.printf "  ESS: %.1f (%.1f%%)\n%!" ess (ess *. 100.0 /. float_of_int n);
      
      (* Setup cumulative weights for sampling *)
      let cum_weights = Array.make n 0.0 in
      cum_weights.(0) <- normalized_weights.(0);
      for i = 1 to n - 1 do
        cum_weights.(i) <- cum_weights.(i-1) +. normalized_weights.(i);
      done;
      
      (* Initialize new parallel samples *)
      let new_particles = Array.make s (Tensor.zeros_like particles.(0)) in
      
      (* Select initial sample for each parallel thread based on weights *)
      for i = 0 to s - 1 do
        let u = Random.float 1.0 in
        let j = ref 0 in
        while !j < n - 1 && cum_weights.(!j) < u do
          incr j
        done;
        new_particles.(i) <- particles.(!j)
      done;
      
      (* Calculate iterations per parallel thread *)
      let samples_per_thread = (n + s - 1) / s in
      
      (* Main sampling and exchange loop *)
      for batch = 0 to samples_per_thread - 1 do
        (* For each parallel chain *)
        for i = 0 to s - 1 do
          (* Create log probability function for this temperature *)
          let log_prob sample =
            let log_prior = log_prior_fn sample in
            let log_likelihood = log_likelihood_fn sample in
            Tensor.(log_prior + (Tensor.of_float0 next_beta * log_likelihood))
          in
          
          (* Metropolis update step *)
          let (updated_sample, accepted) = 
            MCMC.metropolis_step new_particles.(i)
              ~proposal_fn:(fun sample step_size -> 
                proposal_fn sample (Tensor.of_float0 !step_size))
              ~log_prob_fn:log_prob
              ~step_size:(Tensor.of_float0 !step_size)
          in
          
          new_particles.(i) <- updated_sample;
          if accepted then incr acceptance_count;
          incr total_updates;
          
          (* Exchange with a random sample from the previous temperature *)
          if Array.length particles > 0 then
            let j = Random.int (Array.length particles) in
            
            (* Calculate log probabilities for exchange *)
            let log_prob_new_at_next = 
              let log_prior = log_prior_fn new_particles.(i) in
              let log_likelihood = log_likelihood_fn new_particles.(i) in
              Tensor.to_float0_exn Tensor.(log_prior + (Tensor.of_float0 next_beta * log_likelihood))
            in
            
            let log_prob_current_at_current = 
              let log_prior = log_prior_fn particles.(j) in
              let log_likelihood = log_likelihood_fn particles.(j) in
              Tensor.to_float0_exn Tensor.(log_prior + (Tensor.of_float0 !beta * log_likelihood))
            in
            
            let log_prob_new_at_current = 
              let log_prior = log_prior_fn new_particles.(i) in
              let log_likelihood = log_likelihood_fn new_particles.(i) in
              Tensor.to_float0_exn Tensor.(log_prior + (Tensor.of_float0 !beta * log_likelihood))
            in
            
            let log_prob_current_at_next = 
              let log_prior = log_prior_fn particles.(j) in
              let log_likelihood = log_likelihood_fn particles.(j) in
              Tensor.to_float0_exn Tensor.(log_prior + (Tensor.of_float0 next_beta * log_likelihood))
            in
            
            (* Compute exchange acceptance ratio *)
            let log_ratio = (log_prob_current_at_next +. log_prob_new_at_current) -. 
                          (log_prob_new_at_next +. log_prob_current_at_current) in
            
            (* Accept/reject exchange *)
            let u = Random.float 1.0 in
            if log u < log_ratio then begin
              (* Swap samples *)
              let temp = new_particles.(i) in
              new_particles.(i) <- particles.(j);
              particles.(j) <- temp
            end
        done;
        
        (* Adapt step size using Robbins-Monro algorithm *)
        if !total_updates >= 50 then begin
          let acceptance_rate = float_of_int !acceptance_count /. float_of_int !total_updates in
          step_size := update_step_size 
            ~current_step_size:!step_size 
            ~acceptance_rate 
            ~params
            ~iteration:batch;
          
          if verbose then
            Printf.printf "  Step size: %.6f, Acceptance rate: %.4f\n%!" 
              !step_size acceptance_rate;
          
          acceptance_count := 0;
          total_updates := 0
        end
      done;
      
      (* Update particles array with new samples *)
      let new_full_particles = Array.make n (Tensor.zeros_like particles.(0)) in
      
      for i = 0 to s - 1 do
        for j = 0 to samples_per_thread - 1 do
          let idx = i * samples_per_thread + j in
          if idx < n then
            new_full_particles.(idx) <- new_particles.(i)
        done
      done;
      
      Array.blit new_full_particles 0 particles 0 n;
      
      (* Store samples at final temperature *)
      if next_beta >= 1.0 then
        all_samples := Array.to_list particles;
      
      (* Update beta for next iteration *)
      beta := next_beta
    done;
    
    let end_time = Unix.gettimeofday () in
    if verbose then
      Printf.printf "SEMC completed in %.2f seconds with %d temperature steps\n%!" 
        (end_time -. start_time) (List.length !beta_schedule);
    
    (* Return samples and temperature schedule *)
    (!all_samples, List.rev !beta_schedule)
  
  (* Calculate Bayesian free energy *)
  let calculate_free_energy ~samples ~betas ~log_prior_fn ~log_likelihood_fn =
    FreeEnergyCalculation.thermodynamic_integration
      ~samples_at_betas:[samples]
      ~betas:(Array.of_float (Array.of_list betas))
      ~log_likelihood_fn
  
  (* Run multiple chains and check convergence *)
  let run_multiple_chains 
      ~log_prior_fn 
      ~log_likelihood_fn 
      ~proposal_fn 
      ~initial_samples 
      ~n_chains 
      ?(params=default_params)
      ?(r_hat_threshold=1.1)
      ?(verbose=true)
      () =
    
    let n_samples = params.n_samples in
    let n_params = Tensor.shape initial_samples.(0) |> List.hd in
    
    if verbose then
      Printf.printf "Running %d SEMC chains with %d samples each\n" n_chains n_samples;
    
    (* Run each chain in sequence *)
    let chains_results = Array.init n_chains (fun i ->
      if verbose then
        Printf.printf "\nRunning chain %d/%d\n" (i+1) n_chains;
      
      (* Create new initial samples for this chain by adding noise *)
      let chain_initial_samples = Array.map (fun sample ->
        let noise = Tensor.(0.1 * Tensor.randn (Tensor.shape sample) ~device:(Tensor.device sample)) in
        Tensor.(sample + noise)
      ) initial_samples in
      
      (* Run SEMC with less verbose output for chains *)
      run ~log_prior_fn ~log_likelihood_fn ~proposal_fn ~initial_samples:chain_initial_samples
        ~params ~verbose:false ()
    ) in
    
    (* Extract samples and betas from each chain *)
    let all_samples = Array.map fst chains_results in
    let all_betas = Array.map snd chains_results in
    
    (* Check convergence using Gelman-Rubin diagnostic *)
    let (converged, r_hats, max_r_hat) = 
      let r_hats = Array.init n_params (fun i ->
        let extract_param (samples : Tensor.t list) =
          List.map (fun sample ->
            let x = Tensor.get sample [i] |> Tensor.to_float0_exn in
            Tensor.of_float0 x
          ) samples
        in
        
        let param_chains = Array.map (fun chain_samples ->
          extract_param chain_samples
        ) all_samples |> Array.to_list in
        
        Analysis.gelman_rubin_diagnostic param_chains |> Tensor.to_float0_exn
      ) in
      
      let max_r_hat = Array.fold_left max 0.0 r_hats in
      let converged = max_r_hat < r_hat_threshold in
      
      (converged, r_hats, max_r_hat)
    in
    
    if verbose then begin
      Printf.printf "\nConvergence assessment:\n";
      Printf.printf "  Maximum R-hat: %.4f (threshold: %.2f)\n" max_r_hat r_hat_threshold;
      
      for i = 0 to n_params - 1 do
        Printf.printf "  Parameter %d: R-hat = %.4f%s\n" 
          i r_hats.(i) (if r_hats.(i) > r_hat_threshold then " (not converged)" else "");
      done;
      
      Printf.printf "\nOverall convergence: %s\n" (if converged then "Achieved" else "Not achieved");
    end;
    
    (* Combine samples from all chains, discarding burn-in from each *)
    let combined_samples = Array.fold_left (fun acc chain_samples ->
      (* Discard first half as burn-in *)
      let n = List.length chain_samples in
      let burn_in = n / 2 in
      acc @ List.filteri (fun i _ -> i >= burn_in) chain_samples
    ) [] all_samples in
    
    (combined_samples, all_samples, all_betas, r_hats)
end