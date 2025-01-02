open Torch

type schedule = {
  drift : float -> float;
  diffusion : float -> float;
  mean_scale : float -> float;
  var_scale : float -> float;
}

type discretization = float array

let forward_step x t schedule dt noise =
  let f = schedule.drift t in
  let g = schedule.diffusion t in
  let drift_term = Tensor.(x * (f *. dt |> float_value |> of_float)) in
  let diffusion_term = Tensor.(noise * (g *. sqrt dt |> float_value |> of_float)) in
  Tensor.(drift_term + diffusion_term)

let backward_step x t schedule score dt noise =
  let f = schedule.drift t in
  let g = schedule.diffusion t in
  let drift_term = Tensor.(x * (f *. dt |> float_value |> of_float)) in
  let score_term = Tensor.(score x t * ((g *. g *. dt |> float_value) |> of_float)) in
  let diffusion_term = Tensor.(noise * (g *. sqrt dt |> float_value |> of_float)) in
  Tensor.(drift_term - score_term + diffusion_term)

let create_vp_schedule beta_min beta_max =
  let beta t = beta_min +. t *. (beta_max -. beta_min) in
  let alpha t = -.beta t /. 2. in
  {
    drift = (fun t -> alpha t);
    diffusion = (fun t -> sqrt (beta t));
    mean_scale = (fun t -> exp (-0.5 *. beta t *. t));
    var_scale = (fun t -> 1. -. exp (-.beta t *. t));
  }

let stein_divergence p q x =
  let score_p = p x in
  let score_q = q x in
  Tensor.(mean (pow (score_p - score_q) (of_float 2.)))

let local_cost schedule score x t =
  let v = schedule.diffusion t in
  let score_t = score x t in
  let dot_term = Tensor.(sum (score_t * grad_output score_t)) in
  Tensor.((v *. v |> of_float) * dot_term)

let predict x t t_prime schedule score =
  let dt = t_prime -. t in
  let f = schedule.drift t in
  let g = schedule.diffusion t in
  let score_term = score x t in
  let drift_term = Tensor.(x * (f *. dt |> float_value |> of_float)) in
  let correction = Tensor.(score_term * ((g *. g *. dt /. 2.) |> float_value |> of_float)) in
  Tensor.(drift_term - correction)

let correct x t schedule score tau v =
  let rec langevin_step x steps =
    if steps = 0 then x
    else
      let noise = Tensor.randn (Tensor.shape x) in
      let score_term = score x t in
      let dt = tau /. float_of_int steps in
      let x' = Tensor.(x + 
                      (score_term * ((v *. dt) |> float_value |> of_float)) + 
                      (noise * (sqrt (2. *. v *. dt) |> float_value |> of_float))) in
      langevin_step x' (steps - 1)
  in
  langevin_step x 10

let corrector_cost schedule score x t t_prime =
  let v = schedule.diffusion t_prime in
  let score_t = score x t in
  let score_t_prime = score x t_prime in
  Tensor.((v *. v |> of_float) * 
          mean (pow (score_t_prime - score_t) (of_float 2.)))

let predictor_cost schedule score x t t_prime =
  let v = schedule.diffusion t_prime in
  let predicted_x = predict x t t_prime schedule score in
  let score_t_prime = score predicted_x t_prime in
  let predicted_score = score x t in
  Tensor.((v *. v |> of_float) * 
          mean (pow (score_t_prime - predicted_score) (of_float 2.)))

let log_det_lu matrix =
  let lu, pivots = Tensor.lu_with_pivots matrix in
  let diag = Tensor.diag lu in
  Tensor.(sum (log (abs diag)))

let log_det_stochastic matrix n_samples =
  let d = (Tensor.shape matrix).(0) in
  let trace_term = 
    HutchinsonEstimator.adaptive_trace_estimation
      (fun v -> Tensor.(mm (mm matrix v) v))
      (Tensor.eye d)
      1e-6
      n_samples
  in
  Tensor.(log trace_term / (of_float 2.))

module GeometricOptimization = struct
  module PathGenerator = struct
    type t = {
      generator: float -> float;
      derivative: float -> float;
      metric: float -> float;
    }

    let create metric =
      let rec generator s =
        let total_length = metric 1.0 in
        let target_length = s *. total_length in
        
        (* Binary search for time t where metric integral equals target_length *)
        let rec binary_search a b =
          let mid = (a +. b) /. 2.0 in
          let length_mid = metric mid in
          if abs_float (length_mid -. target_length) < 1e-6 then mid
          else if length_mid < target_length then binary_search mid b
          else binary_search a mid
        in
        binary_search 0.0 1.0
      in
      
      let derivative s =
        let eps = 1e-6 in
        (generator (s +. eps) -. generator (s -. eps)) /. (2. *. eps)
      in
      
      { generator; derivative; metric }
  end

  module EnergyFunctional = struct
    let energy_density path_gen schedule score x s =
      let t = path_gen.PathGenerator.generator s in
      let phi_dot = path_gen.PathGenerator.derivative s in
      let delta = local_cost schedule score x t in
      Tensor.(delta * (phi_dot *. phi_dot |> float_value |> of_float))

    let total_energy path_gen schedule score x n_points =
      let ds = 1. /. float_of_int n_points in
      let energy_sum = ref (Tensor.zeros []) in
      
      for i = 0 to n_points - 1 do
        let s = float_of_int i *. ds in
        energy_sum := Tensor.(!energy_sum + energy_density path_gen schedule score x s)
      done;
      
      Tensor.(!energy_sum * (ds |> float_value |> of_float))
  end

  module RiemannianGeometry = struct
    let metric_tensor schedule score x t =
      let v = schedule.diffusion t in
      let score_t = score x t in
      let jacobian = Tensor.grad score_t in
      Tensor.(jacobian * (v *. v |> float_value |> of_float))

    let christoffel_symbols metric x =
      let grad_metric = Tensor.grad metric in
      let inverse_metric = Tensor.inverse metric in
      Tensor.(mm (mm grad_metric inverse_metric) inverse_metric)

    let integrate_geodesic metric_fn x0 v0 schedule n_steps =
      let dt = 1. /. float_of_int n_steps in
      let state = ref (x0, v0) in
      let trajectory = Array.make (n_steps + 1) x0 in
      
      for i = 0 to n_steps - 1 do
        let x, v = !state in
        let t = float_of_int i *. dt in
        
        let g = metric_fn x t in
        let gamma = christoffel_symbols g x in
        
        let new_x = Tensor.(x + v * (dt |> float_value |> of_float)) in
        let new_v = Tensor.(v - mm (mm gamma v) v * (dt |> float_value |> of_float)) in
        
        state := (new_x, new_v);
        trajectory.(i + 1) <- new_x
      done;
      
      trajectory
  end

  module OptimalSchedule = struct
    let compute schedule score x n_points =
      let metric t =
        let tensor_metric = RiemannianGeometry.metric_tensor schedule score x t in
        Tensor.float_value tensor_metric
      in
      
      let path_gen = PathGenerator.create metric in
      
      Array.init n_points (fun i ->
        let s = float_of_int i /. float_of_int (n_points - 1) in
        path_gen.PathGenerator.generator s
      )

    let verify_optimality schedule score x times =
      let n = Array.length times in
      let energies = Array.make (n - 1) 0. in
      
      for i = 0 to n - 2 do
        let path_gen = PathGenerator.create (fun t ->
          let tensor_metric = RiemannianGeometry.metric_tensor schedule score x t in
          Tensor.float_value tensor_metric
        ) in
        
        let energy = EnergyFunctional.total_energy path_gen schedule score x 100 in
        energies.(i) <- Tensor.float_value energy
      done;
      
      let mean_energy = Array.fold_left (+.) 0. energies /. float_of_int (n - 1) in
      Array.for_all (fun e -> abs_float (e -. mean_energy) < 1e-3) energies
  end
end

let calculate_snr schedule t =
  let s = schedule.mean_scale t in
  let sigma = schedule.var_scale t in
  s *. s /. sigma

let generate_snr_schedule schedule n_steps =
  let snr_start = calculate_snr schedule 0. in
  let snr_end = calculate_snr schedule 1. in
  let log_snr_start = log snr_start in
  let log_snr_end = log snr_end in
  
  Array.init n_steps (fun i ->
    let alpha = float_of_int i /. float_of_int (n_steps - 1) in
    let target_log_snr = log_snr_start +. alpha *. (log_snr_end -. log_snr_start) in
    let target_snr = exp target_log_snr in
    
    let rec binary_search left right =
      if right -. left < 1e-6 then left
      else
        let mid = (left +. right) /. 2. in
        let snr_mid = calculate_snr schedule mid in
        if snr_mid > target_snr then
          binary_search mid right
        else
          binary_search left mid
    in
    binary_search 0. 1.
  )

let quality_metric score x t =
  let score_t = score x t in
  Tensor.(mean (pow score_t (of_float 2.)))

let optimize_quality schedule score x initial_times n_steps =
  let n = Array.length initial_times in
  let best_schedule = ref initial_times in
  let best_quality = ref neg_infinity in
  
  for _ = 1 to n_steps do
    let perturbed = Array.map (fun t ->
      let delta = Random.float 0.1 -. 0.05 in
      max 0. (min 1. (t +. delta))
    ) !best_schedule in
    
    let quality = Array.fold_left (fun acc t ->
      acc +. (quality_metric score x t |> Tensor.float_value)
    ) 0. perturbed in
    
    if quality > !best_quality then begin
      best_quality := quality;
      best_schedule := perturbed
    end
  done;
  !best_schedule

let pathwise_kl schedule score x t t_prime =
  let p_t = fun x -> score x t in
  let p_t_prime = fun x -> score x t_prime in
  stein_divergence p_t p_t_prime x

let optimize_pathwise_kl schedule score x initial_times =
  let n = Array.length initial_times in
  let costs = Array.init (n-1) (fun i ->
    let t = initial_times.(i) in
    let t_next = initial_times.(i+1) in
    pathwise_kl schedule score x t t_next |> Tensor.float_value
  ) in
  GeometricOptimization.OptimalSchedule.compute schedule score x n

let fisher_metric schedule score x t =
  let score_t = score x t in
  let jacobian = Tensor.grad score_t in
  let fisher = Tensor.(mm (transpose jacobian) jacobian) in
  fisher

let generate_fisher_schedule schedule score x n_points =
  let compute_integrated_fisher t =
    let metric = fisher_metric schedule score x t in
    Tensor.(trace metric) |> Tensor.float_value
  in
  
  let total_fisher = ref 0. in
  let fisher_points = Array.init 100 (fun i ->
    let t = float_of_int i /. 99. in
    let fisher = compute_integrated_fisher t in
    total_fisher := !total_fisher +. fisher;
    (!total_fisher, t)
  ) in
  
  Array.init n_points (fun i ->
    let target_fisher = !total_fisher *. float_of_int i /. float_of_int (n_points - 1) in
    let rec find_time idx =
      if idx >= Array.length fisher_points - 1 then fisher_points.(idx-1) |> snd
      else
        let (fisher_curr, t_curr) = fisher_points.(idx) in
        let (fisher_next, t_next) = fisher_points.(idx+1) in
        if fisher_curr <= target_fisher && target_fisher <= fisher_next then
          let alpha = (target_fisher -. fisher_curr) /. (fisher_next -. fisher_curr) in
          t_curr +. alpha *. (t_next -. t_curr)
        else
          find_time (idx + 1)
    in
    find_time 1
  )

let antithetic_sampling f x n_samples =
  let shape = Tensor.shape x in
  let sum = ref (Tensor.zeros []) in
  
  for _ = 1 to n_samples do
    let eps = Tensor.randn shape in
    let f_pos = f eps in
    let f_neg = f (Tensor.neg eps) in
    sum := Tensor.(!sum + (f_pos + f_neg) / (of_float 2.))
  done;
  
  Tensor.(!sum / (float_of_int n_samples |> of_float))

let control_variate_sampling f x baseline n_samples =
  let shape = Tensor.shape x in
  let sum = ref (Tensor.zeros []) in
  let baseline_sum = ref (Tensor.zeros []) in
  
  for _ = 1 to n_samples do
    let eps = Tensor.randn shape in
    let f_val = f eps in
    let b_val = baseline eps in
    sum := Tensor.(!sum + (f_val - b_val));
    baseline_sum := Tensor.(!baseline_sum + b_val)
  done;
  
  let cv_term = Tensor.(!sum / (float_of_int n_samples |> of_float)) in
  let baseline_mean = Tensor.(!baseline_sum / (float_of_int n_samples |> of_float)) in
  Tensor.(cv_term + baseline_mean)

let importance_sampling f x proposal n_samples =
  let shape = Tensor.shape x in
  let sum = ref (Tensor.zeros []) in
  let weights_sum = ref 0. in
  
  for _ = 1 to n_samples do
    let sample = proposal () in
    let log_weight = Tensor.(
      sum (pow (sample) (of_float 2.)) / (of_float (-2.)) -
      sum (pow (x) (of_float 2.)) / (of_float (-2.))
    ) |> Tensor.float_value in
    let weight = exp log_weight in
    sum := Tensor.(!sum + (f sample * (weight |> float_value |> of_float)));
    weights_sum := !weights_sum +. weight
  done;
  
  Tensor.(!sum / (!weights_sum |> float_value |> of_float))

module ComparisonFramework = struct
  type approach_result = {
    schedule: float array;
    quality: float;
    computation_time: float;
    memory_usage: int;
  }

  let compare_approaches schedule score x initial_times =
    let approaches = [
      ("SNR", (fun s sc x i -> SNRSchedule.generate_snr_schedule s (Array.length i)));
      ("Quality", (fun s sc x i -> QualityMaximization.optimize_quality s sc x i 100));
      ("PathwiseKL", PathwiseKL.optimize_pathwise_kl);
      ("Fisher", (fun s sc x i -> FisherMetric.generate_fisher_schedule s sc x (Array.length i)));
    ] in
    
    List.map (fun (name, optimize_fn) ->
      let start_time = Unix.gettimeofday () in
      let schedule = optimize_fn schedule score x initial_times in
      let end_time = Unix.gettimeofday () in
      let quality = QualityMaximization.quality_metric score x schedule.(0) 
                   |> Tensor.float_value in
      let memory_usage = Gc.quick_stat () in
      
      (name, {
        schedule;
        quality;
        computation_time = end_time -. start_time;
        memory_usage = memory_usage.Gc.heap_words * 8;
      })
    ) approaches
end

module LinearSchedule = struct
  let create beta_min beta_max =
    let beta t = beta_min +. t *. (beta_max -. beta_min) in
    let alpha t = exp (-. beta t) in
    let alpha_bar t = exp (-. (beta_min +. t *. (beta_max -. beta_min)) *. t /. 2.) in
    
    {
      drift = (fun t -> -. beta t /. 2.);
      diffusion = (fun t -> sqrt (beta t));
      mean_scale = (fun t -> sqrt (alpha t));
      var_scale = (fun t -> 1. -. alpha_bar t);
    }
end

module CosineSchedule = struct
  let create ?(epsilon=0.008) () =
    let f t = cos ((t +. epsilon) /. (1. +. epsilon) *. Float.pi /. 2.) ** 2. in
    let alpha_bar t = f t /. f 0. in
    let alpha t = 
      if t = 0. then 1.
      else alpha_bar t /. alpha_bar (t -. 0.001)
    in
    
    {
      drift = (fun t -> log (alpha t) /. 2.);
      diffusion = (fun t -> sqrt (1. -. alpha t));
      mean_scale = (fun t -> sqrt (alpha_bar t));
      var_scale = (fun t -> 1. -. alpha_bar t);
    }
end

module AdaptiveSchedule = struct
  type adaptive_state = {
    mutable current_schedule: schedule;
    mutable noise_level: float array;
    mutable adaptation_rate: float;
  }

  let init base_schedule n_points =
    {
      current_schedule = base_schedule;
      noise_level = Array.make n_points 1.0;
      adaptation_rate = 0.1;
    }

  let update_noise_levels state score x t stats =
    let idx = int_of_float (t *. float_of_int (Array.length state.noise_level)) in
    let current_noise = state.noise_level.(idx) in
    
    (* Compute optimal noise level based on statistics *)
    let score_norm = Tensor.(mean (pow (score x t) (of_float 2.))) 
                    |> Tensor.float_value in
    let optimal = 1. /. sqrt score_norm in
    let new_noise = current_noise +. 
                   state.adaptation_rate *. (optimal -. current_noise) in
    
    state.noise_level.(idx) <- new_noise;
    
    (* Update schedule with new noise levels *)
    let base_diffusion = state.current_schedule.diffusion in
    {
      state.current_schedule with
      diffusion = (fun t ->
        let idx = int_of_float (t *. float_of_int (Array.length state.noise_level)) in
        base_diffusion t *. state.noise_level.(idx)
      )
    }

  let adapt_schedule state score x batch_size =
    Array.iteri (fun i _ ->
      let t = float_of_int i /. float_of_int (Array.length state.noise_level - 1) in
      let stats = Tensor.{
        mean = mean x;
        std = std x;
        score_norm = mean (pow (score x t) (of_float 2.));
      } in
      state.current_schedule <- update_noise_levels state score x t stats
    ) state.noise_level;
    
    state.current_schedule
end