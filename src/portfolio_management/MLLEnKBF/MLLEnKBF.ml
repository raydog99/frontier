open Torch

(* Kalman-Bucy *)
module KalmanBucy = struct
  (* Model parameters *)
  type model_params = {
    a : Tensor.t;  (* Square dx x dx matrix - drift term *)
    c : Tensor.t;  (* dy x dx matrix - observation matrix *)
    r1 : Tensor.t; (* Square matrix - signal noise covariance *)
    r2 : Tensor.t; (* Square matrix - observation noise covariance *)
    m0 : Tensor.t; (* Initial mean *)
    p0 : Tensor.t; (* Initial covariance *)
  }

  (* Riccati drift function *)
  let ricc params q =
    (* S := C^T * R2^-1 * C *)
    let s = Tensor.(mm (mm (transpose ~dim0:0 ~dim1:1 params.c) (inverse params.r2)) params.c) in
    
    (* Ricc(Q) = AQ + QA^T - QSQ + R, with R = R1 *)
    let aq = Tensor.mm params.a q in
    let qat = Tensor.mm q (Tensor.transpose ~dim0:0 ~dim1:1 params.a) in
    let qsq = Tensor.mm (Tensor.mm q s) q in
    Tensor.(aq + qat - qsq + params.r1)

  (* Continuous-time Kalman-Bucy filter update *)
  let kalman_bucy_update params m p dt y_old y_new =
    (* Compute S = C^T * R2^-1 * C *)
    let s = Tensor.(mm (mm (transpose ~dim0:0 ~dim1:1 params.c) (inverse params.r2)) params.c) in
    
    (* Update covariance using Riccati equation *)
    let dp = ricc params p in
    let p_new = Tensor.(p + dp * (float_of_int dt)) in
    
    (* Kalman gain: P * C^T * R2^-1 *)
    let k = Tensor.(mm (mm p_new (transpose ~dim0:0 ~dim1:1 params.c)) (inverse params.r2)) in
    
    (* Observation increment *)
    let dy = Tensor.(y_new - y_old) in
    
    (* Innovation term: dy - C*M*dt *)
    let innovation = Tensor.(dy - mm params.c (m * (float_of_int dt))) in
    
    (* Update mean *)
    let dm = Tensor.(mm params.a m * (float_of_int dt) + mm k innovation) in
    let m_new = Tensor.(m + dm) in
    
    (m_new, p_new)

  (* Discrete-time Kalman filter update *)
  let kalman_filter params observations n_steps dt =
    (* Initialize with m0 and p0 *)
    let m = params.m0 in
    let p = params.p0 in
    
    (* Run filter for all observations *)
    let rec filter_step idx m p history =
      if idx >= n_steps - 1 then
        List.rev (m :: history)
      else
        let (m_new, p_new) = 
          kalman_bucy_update params m p dt observations.(idx) observations.(idx+1)
        in
        filter_step (idx + 1) m_new p_new (m :: history)
    in
    
    filter_step 0 m p []
    
  (* McKean-Vlasov diffusion process *)
  let mkv_vanilla_step params x m p dt dw y_old y_new =
    (* Kalman gain term *)
    let kalman_gain = Tensor.(mm (mm p (transpose ~dim0:0 ~dim1:1 params.c)) (inverse params.r2)) in
    
    (* Observation noise *)
    let dv = Tensor.(randn (size params.r2) * sqrt (float_of_int dt)) in
    let sqrt_r2 = Tensor.sqrt params.r2 in
    
    (* Innovation term with perturbed observations *)
    let dy = Tensor.(y_new - y_old) in
    let innovation = Tensor.(
      dy - mm params.c (x * float_of_int dt) - mm sqrt_r2 dv
    ) in
    
    (* Process noise *)
    let sqrt_r1 = Tensor.sqrt params.r1 in
    let proc_noise = Tensor.(mm sqrt_r1 (dw * sqrt (float_of_int dt))) in
    
    (* Update *)
    Tensor.(
      x + mm params.a (x * float_of_int dt) + proc_noise + mm kalman_gain innovation
    )
    
  (* McKean-Vlasov diffusion process *)
  let mkv_deterministic_step params x m p dt dw y_old y_new =
    (* Kalman gain term *)
    let kalman_gain = Tensor.(mm (mm p (transpose ~dim0:0 ~dim1:1 params.c)) (inverse params.r2)) in
    
    (* Innovation term without perturbed observations *)
    let dy = Tensor.(y_new - y_old) in
    let innovation = Tensor.(
      dy - mm params.c ((x + m) / (float_of_int 2) * float_of_int dt)
    ) in
    
    (* Process noise *)
    let sqrt_r1 = Tensor.sqrt params.r1 in
    let proc_noise = Tensor.(mm sqrt_r1 (dw * sqrt (float_of_int dt))) in
    
    (* Update *)
    Tensor.(
      x + mm params.a (x * float_of_int dt) + proc_noise + mm kalman_gain innovation
    )
    
  (* McKean-Vlasov diffusion process *)
  let mkv_deterministic_transport_step params x m p dt y_old y_new =
    (* Kalman gain term *)
    let kalman_gain = Tensor.(mm (mm p (transpose ~dim0:0 ~dim1:1 params.c)) (inverse params.r2)) in
    
    (* Innovation term without perturbed observations *)
    let dy = Tensor.(y_new - y_old) in
    let innovation = Tensor.(
      dy - mm params.c ((x + m) / (float_of_int 2) * float_of_int dt)
    ) in
    
    (* Deterministic drift term replacing random process noise *)
    let drift = Tensor.(
      mm params.r1 (mm (inverse p) (x - m) * float_of_int dt)
    ) in
    
    (* Update *)
    Tensor.(
      x + mm params.a (x * float_of_int dt) + drift + mm kalman_gain innovation
    )
end

(* Localization Functions *)
module Localization = struct
  type localization_function =
    | Uniform
    | Triangular
    | GaspariCohn

  (* Uniform localization function ρr(d) = 1[0,r](d) *)
  let uniform_localization d r =
    if d >= 0.0 && d <= r then 1.0 else 0.0
    
  (* Triangular localization function ρr(d) = (1 - d/r)1[0,r](d) *)
  let triangular_localization d r =
    if d >= 0.0 && d <= r then
      1.0 -. (d /. r)
    else 
      0.0
  
  (* Gaspari-Cohn localization function *)
  let gaspari_cohn_localization d r =
    if d = 0.0 then 
      1.0
    else if d >= r then 
      0.0
    else if d < r/.2.0 then
      (* First case: 0 ≤ d < r/2 *)
      -0.25 *. (d/.r)**5.0 +. 
      0.5 *. (d/.r)**4.0 +. 
      0.625 *. (d/.r)**3.0 -. 
      (5.0/.3.0) *. (d/.r)**2.0 +. 
      1.0
    else 
      (* Second case: r/2 ≤ d < r *)
      (1.0/.12.0) *. (d/.r)**5.0 -. 
      0.5 *. (d/.r)**4.0 +. 
      0.625 *. (d/.r)**3.0 +. 
      (5.0/.3.0) *. (d/.r)**2.0 -. 
      5.0 *. (d/.r) +. 
      4.0 -. 
      (2.0/.3.0) *. (r/.d)
  
  (* Select localization function *)
  let get_localization_function = function
    | Uniform -> uniform_localization
    | Triangular -> triangular_localization
    | GaspariCohn -> gaspari_cohn_localization
    
  (* Visualize localization function  *)
  let visualize_localization_function loc_fn r n_points =
    let loc_fn_impl = get_localization_function loc_fn in
    
    (* Compute function values at different distances *)
    let distances = List.init n_points (fun i -> float_of_int i *. r /. float_of_int (n_points - 1)) in
    let values = List.map (fun d -> loc_fn_impl d r) distances in
    
    Printf.printf "Localization function values (r=%.1f):\n" r;
    Printf.printf "Distance | Value\n";
    Printf.printf "----------------\n";
    
    List.iter2
      (fun d v -> Printf.printf "%8.3f | %5.3f\n" d v)
      distances values;
    
    (distances, values)

  (* Apply localization to a covariance matrix using Schur product *)
  let apply_localization cov loc_fn radius =
    let n = Tensor.shape cov in
    let dim = List.hd n in  (* Assuming square matrix *)
    
    (* Create distance matrix - grid distance *)
    let distance_matrix = Tensor.zeros [dim; dim] in
    for i = 0 to dim - 1 do
      for j = 0 to dim - 1 do
        Tensor.set distance_matrix [i; j] (float_of_int (abs (i - j)));
      done;
    done;

    (* Get the localization function *)
    let loc_fn_impl = get_localization_function loc_fn in
    let r = float_of_int radius in
    
    (* Compute localization weights matrix *)
    let loc_weights = Tensor.zeros_like distance_matrix in
    for i = 0 to dim - 1 do
      for j = 0 to dim - 1 do
        let d = Tensor.get distance_matrix [i; j] in
        let weight = loc_fn_impl d r in
        Tensor.set loc_weights [i; j] weight;
      done;
    done;
    
    (* Apply Schur product (element-wise multiplication) *)
    Tensor.(cov * loc_weights)
  
  (* Localized Mean-field Particle Interpretation *)
  let apply_localized_update particles means cov_localized gain innovations dt =
    List.map2
      (fun particle innovation ->
        Tensor.(particle + mm cov_localized gain * innovation * float_of_int dt))
      particles innovations
end

(* Ensemble Kalman-Bucy Filter *)
module EnKBF = struct
  (* Three different EnKBF variants as described in the paper *)
  type enkbf_variant = 
    | Vanilla        (* VEnKBF - with perturbed observations *)
    | Deterministic  (* DEnKBF - without perturbed observations *)
    | DeterministicTransport (* DTEnKBF - completely deterministic *)

  (* Helper function to compute sample mean *)
  let compute_mean ensemble =
    let n = List.length ensemble in
    let sum = List.fold_left Tensor.(+) (List.hd ensemble) (List.tl ensemble) in
    Tensor.(sum / float_of_int n)

  (* Helper function to compute sample covariance *)
  let compute_covariance ensemble mean =
    let n = List.length ensemble in
    let diff_list = List.map (fun x -> Tensor.(x - mean)) ensemble in
    let sum_outer_products = 
      List.fold_left 
        (fun acc diff -> 
          Tensor.(acc + mm diff (transpose ~dim0:0 ~dim1:1 diff)))
        (Tensor.zeros_like (Tensor.mm (List.hd diff_list) 
                           (Tensor.transpose ~dim0:0 ~dim1:1 (List.hd diff_list))))
        diff_list in
    Tensor.(sum_outer_products / float_of_int (n - 1))
  
  (* Vanilla EnKBF *)
  let vanilla_enkbf_step params particle mean cov dt dw dy =
    (* Generate perturbed observation noise *)
    let dv = Tensor.(randn (size params.KalmanBucy.r2)) in
    let sqrt_r2 = Tensor.sqrt params.r2 in
    
    (* Innovation term with perturbed observations *)
    let innovation = Tensor.(
      dy - mm params.c (particle * float_of_int dt) - 
      mm sqrt_r2 (dv * sqrt (float_of_int dt))
    ) in
    
    (* Kalman gain term *)
    let gain = Tensor.(mm (mm cov (transpose ~dim0:0 ~dim1:1 params.c)) (inverse params.r2)) in
    
    (* Update *)
    Tensor.(
      particle + 
      mm params.a (particle * float_of_int dt) + 
      mm (sqrt params.r1) (dw / sqrt (float_of_int dt)) +
      mm gain innovation
    )
  
  (* Deterministic EnKBF *)
  let deterministic_enkbf_step params particle mean cov dt dw dy =
    (* Innovation term without perturbed observations *)
    let innovation = Tensor.(
      dy - mm params.c ((particle + mean) / (float_of_int 2) * float_of_int dt)
    ) in
    
    (* Kalman gain term *)
    let gain = Tensor.(mm (mm cov (transpose ~dim0:0 ~dim1:1 params.c)) (inverse params.r2)) in
    
    (* Update *)
    Tensor.(
      particle + 
      mm params.a (particle * float_of_int dt) + 
      mm (sqrt params.r1) (dw / sqrt (float_of_int dt)) +
      mm gain innovation
    )
  
  (* Deterministic Transport EnKBF *)
  let deterministic_transport_enkbf_step params particle mean cov dt dy =
    (* Innovation term without perturbed observations *)
    let innovation = Tensor.(
      dy - mm params.c ((particle + mean) / (float_of_int 2) * float_of_int dt)
    ) in
    
    (* Kalman gain term *)
    let gain = Tensor.(mm (mm cov (transpose ~dim0:0 ~dim1:1 params.c)) (inverse params.r2)) in
    
    (* Deterministic term replacing random input *)
    let deterministic_term = Tensor.(
      mm params.r1 (mm (inverse cov) (particle - mean) * float_of_int dt)
    ) in
    
    (* Update *)
    Tensor.(
      particle + 
      mm params.a (particle * float_of_int dt) + 
      deterministic_term +
      mm gain innovation
    )

  (* Simulate one step of the EnKBF for a particle based on the selected variant *)
  let enkbf_step params variant ensemble_i ensemble_mean ensemble_cov dt dw dy =
    match variant with
    | Vanilla ->
        vanilla_enkbf_step params ensemble_i ensemble_mean ensemble_cov dt dw dy
        
    | Deterministic ->
        deterministic_enkbf_step params ensemble_i ensemble_mean ensemble_cov dt dw dy
        
    | DeterministicTransport ->
        deterministic_transport_enkbf_step params ensemble_i ensemble_mean ensemble_cov dt dy

  (* Time-discretized version of EnKBF *)
  let discretized_enkbf_step params variant ensemble_i ensemble_mean ensemble_cov dt k y_data =
    let dy = 
      if k + 1 < Array.length y_data then
        Tensor.(y_data.(k+1) - y_data.(k))
      else
        Tensor.zeros_like y_data.(0)
    in
    
    match variant with
    | Vanilla ->
        let dw = Tensor.(randn (size params.KalmanBucy.m0)) in
        let dv = Tensor.(randn (size params.KalmanBucy.r2)) in
        
        let sqrt_r1 = Tensor.sqrt params.r1 in
        let sqrt_r2 = Tensor.sqrt params.r2 in
        
        let innovation = Tensor.(
          dy - 
          mm params.c (ensemble_i * float_of_int dt) - 
          mm sqrt_r2 (dv * sqrt (float_of_int dt))
        ) in
        
        let gain = Tensor.(mm (mm ensemble_cov (transpose ~dim0:0 ~dim1:1 params.c)) 
                              (inverse params.r2)) in
        
        Tensor.(
          ensemble_i + 
          mm params.a (ensemble_i * float_of_int dt) + 
          mm sqrt_r1 (dw * sqrt (float_of_int dt)) +
          mm gain innovation
        )
        
    | Deterministic ->
        let dw = Tensor.(randn (size params.KalmanBucy.m0)) in
        let sqrt_r1 = Tensor.sqrt params.r1 in
        
        let innovation = Tensor.(
          dy - mm params.c ((ensemble_i + ensemble_mean) / float_of_int 2 * float_of_int dt)
        ) in
        
        let gain = Tensor.(mm (mm ensemble_cov (transpose ~dim0:0 ~dim1:1 params.c)) 
                              (inverse params.r2)) in
        
        Tensor.(
          ensemble_i + 
          mm params.a (ensemble_i * float_of_int dt) + 
          mm sqrt_r1 (dw * sqrt (float_of_int dt)) +
          mm gain innovation
        )
        
    | DeterministicTransport ->
        let innovation = Tensor.(
          dy - mm params.c ((ensemble_i + ensemble_mean) / float_of_int 2 * float_of_int dt)
        ) in
        
        let gain = Tensor.(mm (mm ensemble_cov (transpose ~dim0:0 ~dim1:1 params.c)) 
                              (inverse params.r2)) in
        
        let deterministic_term = Tensor.(
          mm params.r1 (mm (inverse ensemble_cov) (ensemble_i - ensemble_mean) * float_of_int dt)
        ) in
        
        Tensor.(
          ensemble_i + 
          mm params.a (ensemble_i * float_of_int dt) + 
          deterministic_term +
          mm gain innovation
        )

  (* Run EnKBF for multiple particles over multiple time steps *)
  let run_enkbf params variant n_particles n_steps dt y_data =
    (* Initialize particles from N(m0, p0) *)
    let init_ensemble = 
      List.init n_particles 
        (fun _ -> 
          let noise = Tensor.(randn (size params.KalmanBucy.m0)) in
          let sqrt_p0 = Tensor.sqrt params.p0 in  (* Assuming p0 is diagonal *)
          Tensor.(params.m0 + mm sqrt_p0 noise))
    in
    
    (* Simulate particles over time *)
    let rec simulate time_idx ensemble history =
      if time_idx >= n_steps then
        List.rev history
      else
        (* Compute ensemble statistics *)
        let mean = compute_mean ensemble in
        let cov = compute_covariance ensemble mean in
        
        (* Update each particle *)
        let new_ensemble = 
          List.map 
            (fun particle -> 
              match variant with
              | DeterministicTransport ->
                  (* No random noise for deterministic transport *)
                  discretized_enkbf_step params variant particle mean cov dt time_idx y_data
              | _ ->
                  (* Generate Brownian motion increment *)
                  let dw = Tensor.(randn (size params.m0)) in
                  discretized_enkbf_step params variant particle mean cov dt time_idx y_data
            )
            ensemble
        in
        
        (* Store mean and continue *)
        simulate (time_idx + 1) new_ensemble (mean :: history)
    in
    
    simulate 0 init_ensemble []
    
  (* Run EnKBF with localized covariance *)
  let run_localized_enkbf params variant n_particles n_steps dt y_data loc_fn loc_radius =
    (* Initialize particles from N(m0, p0) *)
    let init_ensemble = 
      List.init n_particles 
        (fun _ -> 
          let noise = Tensor.(randn (size params.KalmanBucy.m0)) in
          let sqrt_p0 = Tensor.sqrt params.p0 in
          Tensor.(params.m0 + mm sqrt_p0 noise))
    in
    
    (* Simulate particles over time with localization *)
    let rec simulate time_idx ensemble history_means history_covs =
      if time_idx >= n_steps then
        (List.rev history_means, List.rev history_covs)
      else
        (* Compute ensemble statistics *)
        let mean = compute_mean ensemble in
        let raw_cov = compute_covariance ensemble mean in
        
        (* Apply localization to covariance *)
        let cov = Localization.apply_localization raw_cov loc_fn loc_radius in
        
        (* Update each particle with localized covariance *)
        let new_ensemble = 
          List.map 
            (fun particle -> 
              match variant with
              | DeterministicTransport ->
                  discretized_enkbf_step params variant particle mean cov dt time_idx y_data
              | _ ->
                  let dw = Tensor.(randn (size params.m0)) in
                  discretized_enkbf_step params variant particle mean cov dt time_idx y_data
            )
            ensemble
        in
        
        (* Store mean and continue *)
        simulate (time_idx + 1) new_ensemble (mean :: history_means) (cov :: history_covs)
    in
    
    simulate 0 init_ensemble [] []
end

(* Multilevel Monte Carlo *)
module MLMC = struct
  (* Run single level EnKBF with or without localization *)
  let run_single_level params variant n_particles level_l n_steps dt y_data loc_fn loc_radius =
    (* Delta_t for this level *)
    let delta_t = dt * (1 lsl level_l) in
    let effective_steps = n_steps / (1 lsl level_l) in
    
    (* Initialize particles from N(m0, p0) *)
    let init_ensemble = 
      List.init n_particles 
        (fun _ -> 
          let noise = Tensor.(randn (size params.KalmanBucy.m0)) in
          let sqrt_p0 = Tensor.sqrt params.p0 in
          Tensor.(params.m0 + mm sqrt_p0 noise))
    in
    
    (* Simulate particles over time with localization *)
    let rec simulate time_idx ensemble history =
      if time_idx >= effective_steps then
        List.rev history
      else
        (* Compute ensemble statistics *)
        let mean = EnKBF.compute_mean ensemble in
        let raw_cov = EnKBF.compute_covariance ensemble mean in
        
        (* Apply localization to covariance if needed *)
        let cov = 
          if loc_radius > 0 then
            Localization.apply_localization raw_cov loc_fn loc_radius
          else
            raw_cov
        in
        
        (* Get observation increment for this time step *)
        let current_global_idx = time_idx * (1 lsl level_l) in
        let next_global_idx = min (current_global_idx + (1 lsl level_l)) n_steps in
        
        let dy =
          if next_global_idx < n_steps then
            Tensor.(y_data.(next_global_idx) - y_data.(current_global_idx))
          else
            Tensor.zeros_like y_data.(0)
        in
        
        (* Update each particle with correct discretization *)
        let new_ensemble = 
          List.map 
            (fun particle -> 
              match variant with
              | EnKBF.DeterministicTransport ->
                  EnKBF.discretized_enkbf_step 
                    params variant particle mean cov delta_t current_global_idx y_data
              | _ ->
                  let dw = Tensor.(randn (size params.m0)) in
                  EnKBF.discretized_enkbf_step 
                    params variant particle mean cov delta_t current_global_idx y_data
            )
            ensemble
        in
        
        (* Store mean and continue *)
        simulate (time_idx + 1) new_ensemble (mean :: history)
    in
    
    simulate 0 init_ensemble []
    
  (* Run coupled EnKBF for two levels l and l-1 *)
  let run_coupled_levels params variant n_particles level_l n_steps dt y_data loc_fn loc_radius =
    (* Delta_t for levels l and l-1 *)
    let delta_t_fine = dt * (1 lsl level_l) in
    let delta_t_coarse = dt * (1 lsl (level_l - 1)) in
    
    (* Effective number of steps for each level *)
    let n_steps_fine = n_steps / (1 lsl level_l) in
    let n_steps_coarse = n_steps / (1 lsl (level_l - 1)) in
    
    (* Initialize coupled particles (starting with the same values) *)
    let init_particles = 
      List.init n_particles 
        (fun _ -> 
          let noise = Tensor.(randn (size params.KalmanBucy.m0)) in
          let sqrt_p0 = Tensor.sqrt params.p0 in
          Tensor.(params.m0 + mm sqrt_p0 noise))
    in
    
    let init_fine_ensemble = init_particles in
    let init_coarse_ensemble = init_particles in
    
    (* Simulate coupled particles over time with localization *)
    let rec simulate time_idx_coarse fine_ensemble coarse_ensemble history_fine history_coarse =
      if time_idx_coarse >= n_steps_coarse then
        (List.rev history_fine, List.rev history_coarse)
      else
        (* Calculate corresponding indices *)
        let global_idx_coarse = time_idx_coarse * (1 lsl (level_l - 1)) in
        let time_idx_fine = time_idx_coarse * 2 in
        let global_idx_fine = time_idx_fine * (1 lsl level_l) in
        
        (* Compute ensemble statistics for coarse level *)
        let coarse_mean = EnKBF.compute_mean coarse_ensemble in
        let coarse_raw_cov = EnKBF.compute_covariance coarse_ensemble coarse_mean in
        let coarse_cov = 
          if loc_radius > 0 then
            Localization.apply_localization coarse_raw_cov loc_fn loc_radius
          else
            coarse_raw_cov
        in
        
        (* Compute ensemble statistics for fine level *)
        let fine_mean = EnKBF.compute_mean fine_ensemble in
        let fine_raw_cov = EnKBF.compute_covariance fine_ensemble fine_mean in
        let fine_cov = 
          if loc_radius > 0 then
            Localization.apply_localization fine_raw_cov loc_fn loc_radius
          else
            fine_raw_cov
        in
        
        (* Get observation increments *)
        let next_global_idx_coarse = min (global_idx_coarse + (1 lsl (level_l - 1))) n_steps in
        let next_global_idx_fine_1 = min (global_idx_fine + (1 lsl level_l)) n_steps in
        let next_global_idx_fine_2 = min (next_global_idx_fine_1 + (1 lsl level_l)) n_steps in
        
        let dy_coarse =
          if next_global_idx_coarse < n_steps then
            Tensor.(y_data.(next_global_idx_coarse) - y_data.(global_idx_coarse))
          else
            Tensor.zeros_like y_data.(0)
        in
        
        let dy_fine_1 =
          if next_global_idx_fine_1 < n_steps then
            Tensor.(y_data.(next_global_idx_fine_1) - y_data.(global_idx_fine))
          else
            Tensor.zeros_like y_data.(0)
        in
        
        let dy_fine_2 =
          if next_global_idx_fine_2 < n_steps then
            Tensor.(y_data.(next_global_idx_fine_2) - y_data.(next_global_idx_fine_1))
          else
            Tensor.zeros_like y_data.(0)
        in
end


(* Parameter estimation module using MLLEnKBF and normalizing constant estimation *)
module ParameterEstimation = struct
  (* Parameter estimation configuration *)
  type config = {
    params: KalmanBucy.model_params;  (* Initial model parameters *)
    mllenk_config: MLLEnKBF.config;   (* Configuration for MLLEnKBF *)
    param_idx: int list;              (* Indices of parameters to estimate *)
    param_bounds: (float * float) list; (* Min and max bounds for each parameter *)
    n_iterations: int;                (* Number of optimization iterations *)
    learning_rate: float;             (* Learning rate for optimization *)
  }
  
  (* Extract parameter values from model *)
  let extract_params params indices =
    List.map 
      (fun idx ->
        (* Handle different parameter matrices *)
        if idx < Tensor.numel params.KalmanBucy.a then
          (* Parameter in A matrix *)
          let flat_a = Tensor.reshape params.a [Tensor.numel params.a] in
          Tensor.get flat_a [idx]
        else if idx < Tensor.numel params.a + Tensor.numel params.c then
          (* Parameter in C matrix *)
          let adj_idx = idx - Tensor.numel params.a in
          let flat_c = Tensor.reshape params.c [Tensor.numel params.c] in
          Tensor.get flat_c [adj_idx]
        else if idx < Tensor.numel params.a + Tensor.numel params.c + Tensor.numel params.r1 then
          (* Parameter in R1 matrix *)
          let adj_idx = idx - Tensor.numel params.a - Tensor.numel params.c in
          let flat_r1 = Tensor.reshape params.r1 [Tensor.numel params.r1] in
          Tensor.get flat_r1 [adj_idx]
        else
          (* For simplicity, other parameters not supported *)
          0.0
      )
      indices
  
  (* Update model with new parameter values *)
  let update_params params indices values =
    (* Create a copy of the original parameters *)
    let new_params = { 
      KalmanBucy.
      a = Tensor.copy params.a;
      c = Tensor.copy params.c;
      r1 = Tensor.copy params.r1;
      r2 = Tensor.copy params.r2;
      m0 = Tensor.copy params.m0;
      p0 = Tensor.copy params.p0;
    } in
    
    (* Update each parameter *)
    List.iter2
      (fun idx value ->
        if idx < Tensor.numel params.a then
          (* Parameter in A matrix *)
          let flat_a = Tensor.reshape new_params.a [Tensor.numel params.a] in
          Tensor.set flat_a [idx] value;
          (* Reshape back might not be needed for in-place operations *)
        else if idx < Tensor.numel params.a + Tensor.numel params.c then
          (* Parameter in C matrix *)
          let adj_idx = idx - Tensor.numel params.a in
          let flat_c = Tensor.reshape new_params.c [Tensor.numel params.c] in
          Tensor.set flat_c [adj_idx] value;
        else if idx < Tensor.numel params.a + Tensor.numel params.c + Tensor.numel params.r1 then
          (* Parameter in R1 matrix *)
          let adj_idx = idx - Tensor.numel params.a - Tensor.numel params.c in
          let flat_r1 = Tensor.reshape new_params.r1 [Tensor.numel params.r1] in
          Tensor.set flat_r1 [adj_idx] value;
      )
      indices values;
    
    new_params
  
  (* Objective function using log normalizing constant *)
  let log_norm_constant_objective params config observations =
    (* Compute log normalizing constant *)
    let log_nc = 
      MLLEnKBF.run_algorithm2_log params config.mllenk_config observations
      |> Tensor.to_float0_exn
    in
    
    (* Return negative log normalizing constant (for minimization) *)
    -1.0 *. log_nc
  
  (* Gradient-free optimization using Differential Evolution *)
  let differential_evolution config observations =
    let n_params = List.length config.param_idx in
    let population_size = max 10 (4 * n_params) in
    
    (* Initialize population *)
    let init_population =
      List.init population_size
        (fun _ ->
          List.map2
            (fun (min_val, max_val) _ ->
              min_val +. Random.float (max_val -. min_val))
            config.param_bounds config.param_idx)
    in
    
    (* Evaluate fitness of a candidate solution *)
    let evaluate candidate =
      let model = update_params config.params config.param_idx candidate in
      log_norm_constant_objective model config observations
    in
    
    (* Main DE algorithm loop *)
    let rec evolve population best_solution best_fitness iteration =
      if iteration >= config.n_iterations then
        best_solution
      else
        (* Report progress *)
        Printf.printf "Iteration %d/%d, Best fitness: %f\n" 
          iteration config.n_iterations best_fitness;
        
        (* Generate new population *)
        let new_population =
          List.mapi
            (fun i candidate ->
              (* Select three random individuals different from current *)
              let rec select_random_indices excluded n acc =
                if List.length acc >= n then
                  List.take n acc
                else
                  let r = Random.int population_size in
                  if r <> excluded && not (List.mem r acc) then
                    select_random_indices excluded n (r :: acc)
                  else
                    select_random_indices excluded n acc
              in
              
              let [a_idx; b_idx; c_idx] = select_random_indices i 3 [] in
              let a = List.nth population a_idx in
              let b = List.nth population b_idx in
              let c = List.nth population c_idx in
              
              (* Mutation and crossover *)
              let f = 0.8 in  (* Differential weight *)
              let cr = 0.9 in (* Crossover rate *)
              
              let trial =
                List.mapi
                  (fun j x ->
                    let (min_val, max_val) = List.nth config.param_bounds j in
                    if Random.float 1.0 < cr || j = Random.int n_params then
                      (* Mutation *)
                      let v = List.nth a j +. f *. (List.nth b j -. List.nth c j) in
                      (* Bound constraints *)
                      max min_val (min max_val v)
                    else
                      x)
                  candidate
              in
              
              (* Selection *)
              let candidate_fitness = evaluate candidate in
              let trial_fitness = evaluate trial in
              
              if trial_fitness < candidate_fitness then
                trial
              else
                candidate)
            population
        in
        
        (* Find best solution in new population *)
        let (new_best_solution, new_best_fitness) =
          List.fold_left
            (fun (best_sol, best_fit) sol ->
              let fit = evaluate sol in
              if fit < best_fit then
                (sol, fit)
              else
                (best_sol, best_fit))
            (best_solution, best_fitness)
            new_population
        in
        
        (* Continue evolution *)
        evolve new_population new_best_solution new_best_fitness (iteration + 1)
    in
    
    (* Evaluate initial population and find best *)
    let (init_best_solution, init_best_fitness) =
      List.fold_left
        (fun (best_sol, best_fit) sol ->
          let fit = evaluate sol in
          if best_sol = [] || fit < best_fit then
            (sol, fit)
          else
            (best_sol, best_fit))
        ([], Float.infinity)
        init_population
    in
    
    (* Run evolution *)
    let final_solution = evolve init_population init_best_solution init_best_fitness 0 in
    
    (* Return optimized model *)
    update_params config.params config.param_idx final_solution
  
  (* Simple gradient-free optimization using finite differences *)
  let optimize_simple config observations =
    Printf.printf "Starting parameter estimation using simple gradient-free optimization...\n";
    Printf.printf "Parameters to estimate: %d\n" (List.length config.param_idx);
    
    (* Extract initial parameter values *)
    let init_params = extract_params config.params config.param_idx in
    
    (* Objective function: log normalizing constant *)
    let objective_fn params =
      let current_model = update_params config.params config.param_idx params in
      let nc = MLLEnKBF.run_algorithm2 current_model config.mllenk_config observations in
      -1.0 *. (Tensor.log nc |> Tensor.to_float0_exn)
    in
    
    (* Simple coordinate descent optimizer *)
    let rec optimize_iter iteration current_params best_params best_value =
      Printf.printf "Iteration %d/%d\n" iteration config.n_iterations;
      
      if iteration >= config.n_iterations then
        best_params
      else
        (* Evaluate current parameters *)
        let current_value = objective_fn current_params in
        
        Printf.printf "  Current value: %f\n" current_value;
        
        (* Check if this is the best so far *)
        let (new_best_params, new_best_value) =
          if current_value < best_value then
            (current_params, current_value)
          else
            (best_params, best_value)
        in
        
        (* Perturb parameters with random walk *)
        let step_size = config.learning_rate *. (1.0 -. float_of_int iteration /. float_of_int config.n_iterations) in
        
        let new_params =
          List.mapi
            (fun i param ->
              let (min_val, max_val) = List.nth config.param_bounds i in
              let noise = (Random.float 2.0 -. 1.0) *. step_size in
              max min_val (min max_val (param +. noise))
            )
            current_params
        in
        
        optimize_iter (iteration + 1) new_params new_best_params new_best_value
    in
    
    (* Run optimization *)
    let best_params = optimize_iter 0 init_params init_params Float.infinity in
    
    (* Return optimized model *)
    update_params config.params config.param_idx best_params
end