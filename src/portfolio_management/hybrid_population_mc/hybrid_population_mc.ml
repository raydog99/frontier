open Torch

let log_sum_exp x ~dim =
  let max_val = Tensor.max x ~dim:[dim] ~keepdim:true in
  let shifted = Tensor.(x - max_val) in
  Tensor.(max_val + log (sum (exp shifted) ~dim:[dim]))

let compute_covariance samples weights =
  let weighted_mean = Tensor.(sum (samples * weights.unsqueeze ~dim:1) ~dim:[0]) in
  let centered = Tensor.(samples - weighted_mean) in
  Tensor.(
    matmul (transpose centered ~dim0:1 ~dim1:0) 
      (centered * weights.unsqueeze ~dim:1)
  )

let safe_log x = 
  let eps = 1e-10 in
  Tensor.(log (maximum x (Tensor.f eps)))

let stable_softmax x =
  let max_x = Tensor.max x ~dim:[1] ~keepdim:true in
  let shifted = Tensor.(x - max_x) in
  let exp_x = Tensor.exp shifted in
  let sum_exp = Tensor.sum exp_x ~dim:[1] ~keepdim:true in
  Tensor.(exp_x / sum_exp)

let stable_importance_weights log_weights =
  let max_log = Tensor.max log_weights ~dim:[0] ~keepdim:true in
  let shifted = Tensor.(log_weights - max_log) in
  let weights = Tensor.exp shifted in
  let sum_weights = Tensor.sum weights ~dim:[0] ~keepdim:true in
  Tensor.(weights / sum_weights)

module Proposal = struct
  type t = {
    mu: Tensor.t;
    sigma: Tensor.t;
    dim: int;
    log_det_sigma: float;
  }

  let create ~mu ~sigma ~dim =
    let log_det_sigma = 
      Tensor.sum (Tensor.log sigma) 
      |> Tensor.to_float0_exn
    in
    {mu; sigma; dim; log_det_sigma}

  let log_prob prop x =
    let z = Tensor.((x - prop.mu) / prop.sigma) in
    let mahal = Tensor.(sum (z * z) ~dim:[1]) in
    Tensor.(
      (Tensor.f (-0.5 *. Float.of_int prop.dim *. Float.log (2. *. Float.pi))) -
      (Tensor.f prop.log_det_sigma) -
      (Tensor.f 0.5 * mahal)
    )

  let sample prop ~n_samples =
    let noise = Tensor.randn [n_samples; prop.dim] in
    Tensor.(prop.mu + (noise * prop.sigma))
end

module WeightingScheme = struct
  type t = Standard | DeterministicMixture
  
  let compute_dm_weights ~samples ~proposals ~target_log_prob =
    let log_target = target_log_prob samples in
    
    (* Compute proposal mixture denominator *)
    let proposal_log_probs = Array.map (fun prop ->
      Proposal.log_prob prop samples
    ) proposals in
    
    let log_mix_proposal = 
      log_sum_exp 
        (Tensor.stack ~dim:1 (Array.to_list proposal_log_probs))
        ~dim:1
    in
    
    (* Compute and normalize weights *)
    let log_weights = Tensor.(log_target - log_mix_proposal) in
    stable_importance_weights log_weights
end

module ModeTracking = struct
  type mode = {
    location: Tensor.t;
    covariance: Tensor.t;
    weight: float;
    visits: int;
  }

  type mode_state = {
    discovered_modes: mode list;
    mode_distances: float array array option;
    min_mode_distance: float;
  }

  let create_mode_state ~dim =
    let min_mode_distance = Float.sqrt (Float.of_int dim) *. 2. in
    {
      discovered_modes = [];
      mode_distances = None;
      min_mode_distance;
    }

  let compute_mode_distance m1 m2 =
    let diff = Tensor.(m1.location - m2.location) in
    Tensor.(sum (diff * diff)) |> Tensor.to_float0_exn |> Float.sqrt

  let is_new_mode mode existing_modes ~min_distance =
    List.for_all (fun existing ->
      compute_mode_distance mode existing > min_distance
    ) existing_modes

  let update_mode_tracking ~state ~samples ~weights ~target_log_prob =
    let batch_size = Tensor.size samples 0 in
    
    (* Find potential modes from current samples *)
    let potential_modes = 
      List.init batch_size (fun i ->
        let sample = Tensor.select samples 0 i in
        let weight = Tensor.to_float0_exn (Tensor.select weights 0 i) in
        let log_prob = target_log_prob (Tensor.reshape sample ~shape:[1; -1]) in
        {
          location = sample;
          covariance = Tensor.zeros_like sample;
          weight;
          visits = 1;
        }
      )
      |> List.filter (fun mode -> 
           mode.weight > 0.01 *. (float_of_int batch_size)
         )
    in

    (* Update modes list *)
    let updated_modes =
      List.fold_left (fun acc potential_mode ->
        if is_new_mode potential_mode acc ~min_distance:state.min_mode_distance then
          potential_mode :: acc
        else
          let closest_mode = 
            List.find (fun existing ->
              compute_mode_distance potential_mode existing <= 
              state.min_mode_distance
            ) acc
          in
          let merged_mode = {
            location = Tensor.(
              (closest_mode.location * Tensor.f (Float.of_int closest_mode.visits) +
               potential_mode.location) / 
              Tensor.f (Float.of_int (closest_mode.visits + 1))
            );
            covariance = closest_mode.covariance;
            weight = max closest_mode.weight potential_mode.weight;
            visits = closest_mode.visits + 1;
          } in
          merged_mode :: (List.filter (fun m -> m != closest_mode) acc)
      ) state.discovered_modes potential_modes
    in

    { state with discovered_modes = updated_modes }
end

module HMC = struct
  type config = {
    n_leapfrog: int;
    step_size: float;
    mass: Tensor.t;
  }

  let create_config ?(n_leapfrog=10) ?(step_size=0.1) ~dim =
    {
      n_leapfrog;
      step_size;
      mass = Tensor.ones [dim];
    }

  let integrate ~target_log_prob ~initial_pos ~mass ~epsilon ~n_steps =
    let initial_mom = Tensor.(randn (size initial_pos) * sqrt mass) in
    
    let initial_energy = 
      let k = Tensor.(sum (initial_mom * initial_mom / (Tensor.f 2.) / mass)) in
      let u = -.Tensor.to_float0_exn (target_log_prob initial_pos) in
      Tensor.to_float0_exn k +. u
    in

    let rec leapfrog_steps pos mom step_count =
      if step_count = 0 then (pos, initial_energy)
      else
        (* Compute gradient *)
        let pos_grad = Tensor.set_requires_grad pos ~requires_grad:true in
        let log_prob = target_log_prob pos_grad in
        let grad = 
          Tensor.backward log_prob;
          Tensor.grad pos_grad
        in
        
        (* Half step momentum *)
        let mom' = Tensor.(mom + (grad * Tensor.f (epsilon /. 2.))) in
        
        (* Full step position *)
        let pos' = Tensor.(pos + (mom' * Tensor.f epsilon / mass)) in
        
        (* Half step momentum *)
        let pos_grad' = Tensor.set_requires_grad pos' ~requires_grad:true in
        let log_prob' = target_log_prob pos_grad' in
        let grad' = 
          Tensor.backward log_prob';
          Tensor.grad pos_grad'
        in
        let mom'' = Tensor.(mom' + (grad' * Tensor.f (epsilon /. 2.))) in
        
        leapfrog_steps pos' mom'' (step_count - 1)
    in
    
    leapfrog_steps initial_pos initial_mom n_steps

  let sample ~position ~config ~target_log_prob =
    let final_pos, initial_energy = 
      integrate 
        ~target_log_prob 
        ~initial_pos:position 
        ~mass:config.mass 
        ~epsilon:config.step_size 
        ~n_steps:config.n_leapfrog
    in
    
    let final_energy =
      let u = -.Tensor.to_float0_exn (target_log_prob final_pos) in
      u
    in
    
    let accept_prob = Float.min 1.0 (Float.exp (initial_energy -. final_energy)) in
    
    if Random.float 1.0 < accept_prob then
      (final_pos, true)
    else
      (position, false)
end

module Monitor = struct
  type statistics = {
    mean_estimate: Tensor.t;
    covariance_estimate: Tensor.t;
    ess: float;
    log_weights: Tensor.t;
    kl_estimate: float option;
    mode_coverage: float;
  }

  let compute_ess ~weights =
    let normalized = 
      let sum = Tensor.sum weights ~dim:[0] ~keepdim:true in
      Tensor.(weights / sum)
    in
    let squared_sum = Tensor.(sum (normalized * normalized)) in
    1. /. (Tensor.to_float0_exn squared_sum)

  let compute_statistics ~samples ~weights ~true_mean_opt =
    let mean_est = 
      Tensor.(sum (samples * weights.unsqueeze ~dim:1) ~dim:[0])
    in
    let cov_est = compute_covariance samples weights in
    
    let log_weights = 
      let max_weight = Tensor.max weights ~dim:[0] ~keepdim:true in
      Tensor.(log weights - log max_weight)
    in
    
    let kl_est = match true_mean_opt with
      | Some true_mean ->
          let diff = Tensor.(mean_est - true_mean) in
          Some (Tensor.(sum (diff * diff)) |> Tensor.to_float0_exn)
      | None -> None
    in
    
    {
      mean_estimate = mean_est;
      covariance_estimate = cov_est;
      ess = compute_ess ~weights;
      log_weights;
      kl_estimate = kl_est;
      mode_coverage = 0.0; 
    }
end

module CurvedAdaptation = struct
  type metric_type = 
    | Euclidean 
    | Riemannian of {epsilon: float; decay: float}

  let compute_riemannian_metric ~pos ~grad ~epsilon =
    let norm_grad = Tensor.(sqrt (sum (grad * grad) ~dim:[1])) in
    Tensor.(ones_like norm_grad + (norm_grad * Tensor.f epsilon))

  let curved_leapfrog_step ~pos ~mom ~target_log_prob ~mass ~epsilon ~metric_type =
    let pos_grad = Tensor.set_requires_grad pos ~requires_grad:true in
    let log_prob = target_log_prob pos_grad in
    let grad = 
      Tensor.backward log_prob;
      Tensor.grad pos_grad
    in
    
    let metric = match metric_type with
      | Euclidean -> Tensor.ones_like mom
      | Riemannian {epsilon; _} -> 
          compute_riemannian_metric ~pos ~grad ~epsilon
    in
    
    let mom_half = Tensor.(mom + (grad * Tensor.f (epsilon /. 2.) * metric)) in
    let pos_new = Tensor.(pos + (mom_half * Tensor.f epsilon / mass * metric)) in
    
    let pos_new_grad = Tensor.set_requires_grad pos_new ~requires_grad:true in
    let log_prob_new = target_log_prob pos_new_grad in
    let grad_new = 
      Tensor.backward log_prob_new;
      Tensor.grad pos_new_grad
    in
    
    let metric_new = match metric_type with
      | Euclidean -> metric
      | Riemannian {epsilon; _} ->
          compute_riemannian_metric ~pos:pos_new ~grad:grad_new ~epsilon
    in
    
    let mom_new = Tensor.(mom_half + (grad_new * Tensor.f (epsilon /. 2.) * metric_new)) in
    
    pos_new, mom_new

  let generate_preliminary_locations ~samples ~weights ~config =
    let dim = Tensor.size samples 1 in
    let metric_type = 
      if dim > 20 then
        Riemannian {epsilon = 0.01; decay = 0.95}
      else
        Euclidean
    in
    
    let n_locations = config.HPMC.n_proposals in
    
    (* Initialize from weighted samples *)
    let initial_locs = Array.init n_locations (fun i ->
      let idx = 
        let cum_weights = Tensor.cumsum weights ~dim:0 in
        let u = Random.float (Tensor.to_float0_exn (Tensor.select cum_weights (-1) 0)) in
        let rec find_idx i =
          if i >= Tensor.size weights 0 then i - 1
          else if Tensor.to_float0_exn (Tensor.select cum_weights i 0) > u then i
          else find_idx (i + 1)
        in
        find_idx 0
      in
      Tensor.select samples idx 0
    ) in
    
    (* Apply curved HMC steps *)
    Array.map (fun loc ->
      let mass = Tensor.ones [dim] in
      let mom = Tensor.(randn [1; dim] * sqrt mass) in
      
      let rec hmc_steps pos mom n =
        if n = 0 then pos
        else
          let pos_new, mom_new = 
            curved_leapfrog_step 
              ~pos ~mom 
              ~target_log_prob:config.target_log_prob
              ~mass 
              ~epsilon:config.step_size 
              ~metric_type
          in
          hmc_steps pos_new mom_new (n-1)
      in
      
      hmc_steps loc mom config.n_leapfrog
    ) initial_locs
end

module AdaptiveScheduler = struct
  type phase = {
    exploration_weight: float;
    refinement_weight: float;
    step_size: float;
    n_leapfrog: int;
  }

  type schedule = {
    phases: phase array;
    current_phase: int;
    phase_length: int;
    total_iterations: int;
  }

  let create_schedule ~n_iterations ~initial_step_size ~n_leapfrog =
    let n_phases = 4 in
    let phase_length = n_iterations / n_phases in
    {
      phases = Array.init n_phases (fun i ->
        let t = float_of_int i /. float_of_int (n_phases - 1) in
        {
          exploration_weight = 1.0 -. t;
          refinement_weight = t;
          step_size = initial_step_size *. (1.0 -. 0.5 *. t);
          n_leapfrog;
        }
      );
      current_phase = 0;
      phase_length;
      total_iterations = n_iterations;
    }

  let get_current_phase schedule iteration =
    let phase_idx = min 
      (iteration / schedule.phase_length)
      (Array.length schedule.phases - 1)
    in
    schedule.phases.(phase_idx)
end

module IntegratedAdaptation = struct
  type state = {
    phase: [`Exploration | `Refinement | `Mixed];
    step_size: float;
    n_leapfrog: int;
    metric: CurvedAdaptation.metric_type;
    mode_state: ModeTracking.mode_state;
    adaptation_weights: float array;
    accepted_moves: int;
  }

  let create_state config =
    {
      phase = `Exploration;
      step_size = config.HPMC.step_size;
      n_leapfrog = config.HPMC.n_leapfrog;
      metric = if config.dim > 20 then 
                 CurvedAdaptation.Riemannian {epsilon = 0.01; decay = 0.95}
               else CurvedAdaptation.Euclidean;
      mode_state = ModeTracking.create_mode_state ~dim:config.dim;
      adaptation_weights = [|0.5; 0.3; 0.2|];
      accepted_moves = 0;
    }

  let adapt_step ~state ~config ~samples ~weights ~target_log_prob =
    let mode_state = ModeTracking.update_mode_tracking 
      ~state:state.mode_state 
      ~samples ~weights 
      ~target_log_prob 
    in
    
    let curved_proposals = 
      CurvedAdaptation.generate_preliminary_locations 
        ~samples ~weights ~config
    in
    
    let mode_proposals =
      Array.of_list (List.map (fun mode ->
        mode.ModeTracking.location
      ) mode_state.discovered_modes)
    in
    
    let new_phase =
      let ess = Monitor.compute_ess ~weights in
      let n_modes = List.length mode_state.discovered_modes in
      if ess < 0.3 *. float_of_int (Tensor.size samples 0) || n_modes < 2 then
        `Exploration
      else if ess > 0.7 *. float_of_int (Tensor.size samples 0) then
        `Refinement
      else
        `Mixed
    in
    
    let new_weights = match new_phase with
      | `Exploration -> [|0.6; 0.2; 0.2|]
      | `Refinement -> [|0.2; 0.6; 0.2|]
      | `Mixed -> [|0.4; 0.4; 0.2|]
    in
    
    let n_curved = int_of_float (new_weights.(0) *. float_of_int config.n_proposals) in
    let n_mode = int_of_float (new_weights.(1) *. float_of_int config.n_proposals) in
    let n_local = config.n_proposals - n_curved - n_mode in
    
    let combined_proposals = Array.concat [
      Array.sub curved_proposals 0 n_curved;
      (if Array.length mode_proposals > 0 then 
         Array.sub mode_proposals 0 (min n_mode (Array.length mode_proposals))
       else [||]);
      Array.sub curved_proposals n_curved n_local;
    ] in
    
    { state with
      phase = new_phase;
      adaptation_weights = new_weights;
      mode_state;
    }, combined_proposals

end

module HPMC = struct
  type config = {
    n_proposals: int;
    n_samples: int;
    n_iterations: int;
    dim: int;
    step_size: float;
    n_leapfrog: int;
  }

  type state = {
    samples: Tensor.t;
    weights: Tensor.t;
    proposals: Proposal.t array;
    integrated_adaptation: IntegratedAdaptation.state;
    iteration: int;
    mode_tracking: ModeTracking.mode_state;
    stats: Monitor.statistics;
  }

  let create_config ~n_proposals ~n_samples ~n_iterations ~dim =
    {
      n_proposals;
      n_samples;
      n_iterations;
      dim;
      step_size = 0.1;
      n_leapfrog = 10;
    }

  let init_state config =
    let proposals = Array.init config.n_proposals (fun _ ->
      let mu = Tensor.(randn [1; config.dim] * Tensor.f 4.) in
      let sigma = Tensor.ones [1; config.dim] in
      Proposal.create ~mu ~sigma ~dim:config.dim
    ) in
    {
      samples = Tensor.zeros [config.n_proposals * config.n_samples; config.dim];
      weights = Tensor.zeros [config.n_proposals * config.n_samples];
      proposals;
      integrated_adaptation = IntegratedAdaptation.create_state config;
      iteration = 0;
      mode_tracking = ModeTracking.create_mode_state ~dim:config.dim;
      stats = Monitor.compute_statistics 
        ~samples:Tensor.empty [] 
        ~weights:Tensor.empty [] 
        ~true_mean_opt:None;
    }

  let run config target_log_prob =
    let rec iterate state =
      if state.iteration >= config.n_iterations then
        state
      else
        (* Generate samples *)
        let samples = Array.map (fun prop ->
          Proposal.sample prop ~n_samples:config.n_samples
        ) state.proposals in
        let samples = Tensor.cat ~dim:0 (Array.to_list samples) in
        
        (* Compute weights *)
        let weights = WeightingScheme.compute_dm_weights 
          ~samples 
          ~proposals:state.proposals 
          ~target_log_prob 
        in
        
        (* Update adaptation *)
        let new_adaptation_state, new_proposals = 
          IntegratedAdaptation.adapt_step
            ~state:state.integrated_adaptation
            ~config
            ~samples
            ~weights
            ~target_log_prob
        in
        
        (* Update proposals *)
        let proposals = Array.mapi (fun i _ ->
          let mu = new_proposals.(i) in
          let sigma = Tensor.ones [1; config.dim] in
          Proposal.create ~mu ~sigma ~dim:config.dim
        ) state.proposals in
        
        (* Update statistics *)
        let stats = Monitor.compute_statistics 
          ~samples 
          ~weights 
          ~true_mean_opt:None 
        in
        
        iterate {
          state with
          samples;
          weights;
          proposals;
          integrated_adaptation = new_adaptation_state;
          iteration = state.iteration + 1;
          stats;
        }
    in
    
    iterate (init_state config)
end

module Benchmark = struct
  type distribution_type = 
    | MultiModal2D
    | HighDimBimodal of int
    | BananaShaped of {dim: int; b: float}
    
  type result = {
    ess_history: float array;
    mode_discovery: int array;
    mean_error: float array;
    runtime: float;
    target_evals: int;
    acceptance_rates: float array;
  }

  let create_target = function
    | MultiModal2D -> 
        let means = [|
          [|-10.; -10.|]; [|0.; 16.|]; [|13.; 8.|]; 
          [|9.; 7.|]; [|14.; 14.|]
        |] |> Array.map (fun v -> Tensor.of_float1 v) in
        
        let covs = [|
          Tensor.of_float2 [|[|2.0; 0.6|]; [|0.6; 1.0|]|];
          Tensor.of_float2 [|[|2.0; 0.4|]; [|0.4; 2.0|]|];
          Tensor.of_float2 [|[|2.0; 0.8|]; [|0.8; 2.0|]|];
          Tensor.of_float2 [|[|3.0; 0.0|]; [|0.0; 0.5|]|];
          Tensor.of_float2 [|[|2.0; 0.1|]; [|0.1; 2.0|]|];
        |] in
        
        let target x = 
          let components = Array.mapi (fun i mu ->
            let z = Tensor.(x - mu) in
            let cov_inv = Tensor.inverse covs.(i) in
            Tensor.(
              exp (
                matmul (matmul z cov_inv) (transpose z ~dim0:1 ~dim1:0) * 
                Tensor.f (-0.5)
              ) * 
              Tensor.f (1. /. 5.)
            )
          ) means in
          Tensor.(sum (stack ~dim:0 (Array.to_list components)) ~dim:[0])
        in
        target, means

    | HighDimBimodal dim ->
        let mu1 = Tensor.ones [dim] |> Tensor.mul_scalar 8. in
        let mu2 = Tensor.ones [dim] |> Tensor.mul_scalar (-8.) in
        let sigma = Tensor.ones [dim] |> Tensor.mul_scalar 5. in
        
        let target x =
          let z1 = Tensor.(x - mu1) in
          let z2 = Tensor.(x - mu2) in
          Tensor.(
            (exp (sum (z1 * z1) ~dim:[1] * Tensor.f (-0.5)) +
             exp (sum (z2 * z2) ~dim:[1] * Tensor.f (-0.5))) *
            Tensor.f 0.5
          )
        in
        target, [|mu1; mu2|]

    | BananaShaped {dim; b} ->
        let transform x =
          let x1 = Tensor.select x ~dim:1 ~index:0 in
          let x2 = Tensor.select x ~dim:1 ~index:1 in
          let transformed_x2 = Tensor.(x2 - (Tensor.f b * (x1 * x1))) in
          transformed_x2
        in
        
        let target x =
          let z1 = Tensor.select x ~dim:1 ~index:0 in
          let z2 = transform x in
          let remaining = 
            if dim > 2 then
              let others = Tensor.narrow x ~dim:1 ~start:2 ~length:(dim-2) in
              Tensor.(sum (others * others) ~dim:[1])
            else 
              Tensor.zeros [Tensor.size x 0]
          in
          Tensor.(
            exp ((z1 * z1 + z2 * z2 + remaining) * Tensor.f (-0.5)) *
            Tensor.f (1. /. (Float.sqrt (2. *. Float.pi)))
          )
        in
        let true_mean = Tensor.zeros [dim] in
        target, [|true_mean|]

  let run_benchmark ?(n_repeats=10) dist_type config =
    let target, true_means = create_target dist_type in
    
    Array.init n_repeats (fun _ ->
      let t_start = Unix.gettimeofday () in
      
      let state = HPMC.run config target in
      
      let t_end = Unix.gettimeofday () in
      
      {
        ess_history = Array.make config.n_iterations state.stats.Monitor.ess;
        mode_discovery = Array.init config.n_iterations (fun _ ->
          List.length state.mode_tracking.discovered_modes
        );
        mean_error = Array.init config.n_iterations (fun _ ->
          match state.stats.Monitor.kl_estimate with
          | Some kl -> kl
          | None -> 0.
        );
        runtime = t_end -. t_start;
        target_evals = 
          config.n_iterations * config.n_proposals * config.n_samples;
        acceptance_rates = Array.init config.n_iterations (fun _ ->
          float_of_int state.integrated_adaptation.accepted_moves /
          float_of_int (state.iteration + 1)
        );
      }
    )
end

module Diagnostics = struct
  type diagnostic_report = {
    ess_trend: float array;
    mode_stability: float;
    weight_entropy: float;
    grad_norm_stats: float * float;
    proposal_spread: float;
  }

  let compute_weight_entropy weights =
    let norm_weights = 
      let sum = Tensor.sum weights ~dim:[0] ~keepdim:true in
      Tensor.(weights / sum)
    in
    Tensor.(
      sum (norm_weights * log norm_weights) ~dim:[0] 
      |> neg |> to_float0_exn
    )

  let compute_proposal_spread proposals =
    let locs = Array.map (fun p -> p.Proposal.mu) proposals in
    let stacked = Tensor.stack ~dim:0 (Array.to_list locs) in
    let mean = Tensor.mean stacked ~dim:[0] ~keepdim:true in
    let centered = Tensor.(stacked - mean) in
    Tensor.(mean (sum (centered * centered) ~dim:[1])) 
    |> Tensor.to_float0_exn

  let generate_report state =
    let ess_trend = Array.init 10 (fun i ->
      let idx = max 0 (state.HPMC.iteration - 10 + i) in
      state.stats.Monitor.ess
    ) in
    
    let mode_stability =
      let recent_modes = Array.init 20 (fun i ->
        let idx = max 0 (state.iteration - 20 + i) in
        float_of_int (List.length state.mode_tracking.discovered_modes)
      ) in
      let mean = Array.fold_left (+.) 0. recent_modes /.
                 float_of_int (Array.length recent_modes) in
      let var = Array.fold_left (fun acc x ->
        acc +. (x -. mean) ** 2.
      ) 0. recent_modes in
      var /. float_of_int (Array.length recent_modes)
    in
    
    {
      ess_trend;
      mode_stability;
      weight_entropy = compute_weight_entropy state.weights;
      grad_norm_stats = (0., 0.);  (* Computed from stored gradients if needed *)
      proposal_spread = compute_proposal_spread state.proposals;
    }
end