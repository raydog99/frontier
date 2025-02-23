open Torch

type state = Tensor.t
type observation = Tensor.t
type time = float

type parameters = {
  mu: float;      (* Drift/mean parameter *)
  theta: float;   (* Mean reversion level *)
  sigma: float;   (* Volatility parameter *)
  rho: float;     (* Correlation parameter *)
  kappa: float;   (* Mean reversion speed *)
  xi: float;      (* Volatility of volatility *)
}

(* Statistical functions *)
let normal_pdf mean std x =
  let pi = Tensor.scalar_float Float.pi in
  let exponent = -0.5 *. ((x -. mean) /. std) ** 2. in
  exp exponent /. (std *. sqrt (2. *. pi))

let log_likelihood preds actuals noise_std =
  let n = float_of_int (Array.length actuals) in
  let sum_sq_err = ref 0. in
  Array.iteri (fun i pred ->
    sum_sq_err := !sum_sq_err +. 
      ((actuals.(i) -. pred) /. noise_std) ** 2.
  ) preds;
  -0.5 *. (!sum_sq_err +. n *. log (2. *. Float.pi *. noise_std ** 2.))

(* Random number generation *)
let random_gaussian () =
  let u1 = Random.float 1.0 in
  let u2 = Random.float 1.0 in
  sqrt (-2. *. log u1) *. cos (2. *. Float.pi *. u2)

let random_gamma shape scale =
  let d = shape -. 1.0 /. 3. in
  let c = 1.0 /. sqrt (9. *. d) in
  let rec loop () =
    let x = random_gaussian () in
    let v = 1.0 +. c *. x in
    if v <= 0.0 then loop ()
    else
      let v3 = v ** 3. in
      let u = Random.float 1.0 in
      if u < 1.0 -. 0.0331 *. x ** 4. then
        d *. v3 *. scale
      else if log u < 0.5 *. x ** 2. +. d *. (1.0 -. v3 +. log v3) then
        d *. v3 *. scale
      else
        loop ()
  in loop ()

(* Matrix operations *)
let safe_cholesky mat =
  let n = Tensor.shape mat |> List.hd in
  let l = Tensor.zeros [n; n] in
  
  for i = 0 to n - 1 do
    for j = 0 to i do
      let sum = ref 0. in
      if i = j then begin
        for k = 0 to j - 1 do
          sum := !sum +. Tensor.get l [j; k] ** 2.
        done;
        let val_ = Tensor.get mat [i; i] -. !sum in
        Tensor.set l [i; i] (sqrt val_)
      end else begin
        for k = 0 to j - 1 do
          sum := !sum +. 
            Tensor.get l [i; k] *. Tensor.get l [j; k]
        done;
        Tensor.set l [i; j] 
          ((Tensor.get mat [i; j] -. !sum) /. Tensor.get l [j; j])
      end
    done
  done;
  l

(* Kalman Filter *)
module KalmanFilter = struct
  type t = {
    state: state;
    covariance: Tensor.t;
    transition_matrix: Tensor.t;
    observation_matrix: Tensor.t;
    process_noise: Tensor.t;
    observation_noise: Tensor.t;
  }

  let create ~init_state ~init_cov ~trans_mat ~obs_mat ~proc_noise ~obs_noise = {
    state = init_state;
    covariance = init_cov;
    transition_matrix = trans_mat;
    observation_matrix = obs_mat;
    process_noise = proc_noise;
    observation_noise = obs_noise;
  }

  let predict kf =
    let predicted_state = Tensor.(matmul kf.transition_matrix kf.state) in
    let predicted_cov = Tensor.(
      add (matmul (matmul kf.transition_matrix kf.covariance)
                  (transpose kf.transition_matrix ~dim0:0 ~dim1:1))
          kf.process_noise
    ) in
    { kf with 
      state = predicted_state;
      covariance = predicted_cov }

  let update kf observation =
    let innovation = Tensor.(
      sub observation (matmul kf.observation_matrix kf.state)
    ) in
    let innovation_cov = Tensor.(
      add (matmul (matmul kf.observation_matrix kf.covariance)
                  (transpose kf.observation_matrix ~dim0:0 ~dim1:1))
          kf.observation_noise
    ) in
    let k = Tensor.(
      matmul (matmul kf.covariance 
                     (transpose kf.observation_matrix ~dim0:0 ~dim1:1))
             (inverse innovation_cov)
    ) in
    let updated_state = Tensor.(
      add kf.state (matmul k innovation)
    ) in
    let updated_cov = Tensor.(
      sub kf.covariance
          (matmul (matmul k kf.observation_matrix) kf.covariance)
    ) in
    { kf with 
      state = updated_state;
      covariance = updated_cov }
end

(* Extended Kalman Filter *)
module ExtendedKalmanFilter = struct
  type t = {
    state: state;
    covariance: Tensor.t;
    process_noise: Tensor.t;
    observation_noise: Tensor.t;
    state_transition: state -> state;
    observation_func: state -> observation;
    state_jacobian: state -> Tensor.t;
    observation_jacobian: state -> Tensor.t;
  }

  let create ~init_state ~init_cov ~proc_noise ~obs_noise
            ~state_trans ~obs_func ~state_jac ~obs_jac = {
    state = init_state;
    covariance = init_cov;
    process_noise = proc_noise;
    observation_noise = obs_noise;
    state_transition = state_trans;
    observation_func = obs_func;
    state_jacobian = state_jac;
    observation_jacobian = obs_jac;
  }

  let predict ekf =
    let predicted_state = ekf.state_transition ekf.state in
    let f_jac = ekf.state_jacobian ekf.state in
    let predicted_cov = Tensor.(
      add (matmul (matmul f_jac ekf.covariance)
                  (transpose f_jac ~dim0:0 ~dim1:1))
          ekf.process_noise
    ) in
    { ekf with 
      state = predicted_state;
      covariance = predicted_cov }

  let update ekf observation =
    let h_jac = ekf.observation_jacobian ekf.state in
    let predicted_obs = ekf.observation_func ekf.state in
    
    let innovation = Tensor.(sub observation predicted_obs) in
    let innovation_cov = Tensor.(
      add (matmul (matmul h_jac ekf.covariance)
                  (transpose h_jac ~dim0:0 ~dim1:1))
          ekf.observation_noise
    ) in
    let k = Tensor.(
      matmul (matmul ekf.covariance 
                     (transpose h_jac ~dim0:0 ~dim1:1))
             (inverse innovation_cov)
    ) in
    let updated_state = Tensor.(
      add ekf.state (matmul k innovation)
    ) in
    let updated_cov = Tensor.(
      sub ekf.covariance
          (matmul (matmul k h_jac) ekf.covariance)
    ) in
    { ekf with 
      state = updated_state;
      covariance = updated_cov }

  (* Second-order EKF update *)
  let update_second_order ekf observation =
    let h_jac = ekf.observation_jacobian ekf.state in
    let predicted_obs = ekf.observation_func ekf.state in
    
    (* Compute Hessian *)
    let eps = 1e-6 in
    let state_dim = Tensor.shape ekf.state |> List.hd in
    let hessian = Tensor.zeros [state_dim; state_dim] in
    
    for i = 0 to state_dim - 1 do
      for j = 0 to state_dim - 1 do
        let state_plus_i = Tensor.(add ekf.state 
          (mul_scalar (one_hot state_dim i) eps)) in
        let state_plus_j = Tensor.(add ekf.state 
          (mul_scalar (one_hot state_dim j) eps)) in
        let state_plus_both = Tensor.(add state_plus_i 
          (mul_scalar (one_hot state_dim j) eps)) in
        
        let f_plus_i = ekf.observation_func state_plus_i in
        let f_plus_j = ekf.observation_func state_plus_j in
        let f_plus_both = ekf.observation_func state_plus_both in
        let f = predicted_obs in
        
        Tensor.set hessian [i; j]
          Tensor.(to_float0_exn (
            div (sub (add f_plus_both (neg (add f_plus_i f_plus_j))) f)
                (mul_scalar (ones [1]) (eps *. eps))))
      done
    done;
    
    (* Second-order correction *)
    let correction = Tensor.(
      mul_scalar (matmul hessian ekf.covariance) 0.5
    ) in
    
    (* Standard EKF update with correction *)
    let innovation = Tensor.(sub observation 
      (add predicted_obs correction)) in
    
    let innovation_cov = Tensor.(
      add (matmul (matmul h_jac ekf.covariance)
                  (transpose h_jac ~dim0:0 ~dim1:1))
          ekf.observation_noise
    ) in
    
    let k = Tensor.(
      matmul (matmul ekf.covariance 
                     (transpose h_jac ~dim0:0 ~dim1:1))
             (inverse innovation_cov)
    ) in
    
    let updated_state = Tensor.(
      add ekf.state (matmul k innovation)
    ) in
    let updated_cov = Tensor.(
      sub ekf.covariance
          (matmul (matmul k h_jac) ekf.covariance)
    ) in
    
    { ekf with 
      state = updated_state;
      covariance = updated_cov }
end

(* Square Root Kalman Filter *)
module SquareRootKalmanFilter = struct
  type t = {
    state: state;
    sqrt_covariance: Tensor.t;  (* Cholesky factor of covariance *)
    transition_matrix: Tensor.t;
    observation_matrix: Tensor.t;
    sqrt_process_noise: Tensor.t;  (* Cholesky factor of process noise *)
    sqrt_observation_noise: Tensor.t;  (* Cholesky factor of observation noise *)
  }

  let create ~init_state ~init_cov ~trans_mat ~obs_mat ~proc_noise ~obs_noise =
    let sqrt_cov = Utils.safe_cholesky init_cov in
    let sqrt_proc = Utils.safe_cholesky proc_noise in
    let sqrt_obs = Utils.safe_cholesky obs_noise in
    {
      state = init_state;
      sqrt_covariance = sqrt_cov;
      transition_matrix = trans_mat;
      observation_matrix = obs_mat;
      sqrt_process_noise = sqrt_proc;
      sqrt_observation_noise = sqrt_obs
    }

  let predict kf =
    let n = Tensor.shape kf.state |> List.hd in
    
    (* Form augmented matrix for prediction *)
    let pred_state = Tensor.(matmul kf.transition_matrix kf.state) in
    
    let aug_cov = Tensor.(
      let s1 = matmul kf.transition_matrix kf.sqrt_covariance in
      let s2 = kf.sqrt_process_noise in
      cat [s1; s2] ~dim:1
    ) in
    
    (* QR decomposition *)
    let q, r = Tensor.qr aug_cov in
    let new_sqrt_cov = Tensor.(narrow r ~dim:0 ~start:0 ~length:n) in
    
    { kf with 
      state = pred_state;
      sqrt_covariance = new_sqrt_cov }

  let update kf observation =
    let n = Tensor.shape kf.state |> List.hd in
    
    (* Form innovation system *)
    let pred_obs = Tensor.(matmul kf.observation_matrix kf.state) in
    let innovation = Tensor.(sub observation pred_obs) in
    
    let aug_innov = Tensor.(
      let s1 = matmul kf.observation_matrix kf.sqrt_covariance in
      let s2 = kf.sqrt_observation_noise in
      cat [s1; s2] ~dim:1
    ) in
    
    (* QR decomposition of innovation system *)
    let q, r = Tensor.qr aug_innov in
    let sqrt_innov_cov = r in
    
    (* Solve for Kalman gain using triangular solve *)
    let k = Tensor.(triangular_solve sqrt_innov_cov innovation ~upper:true) in
    
    (* Update state and sqrt covariance *)
    let updated_state = Tensor.(add kf.state (matmul k innovation)) in
    let i_kr = Tensor.(
      sub (eye n)
          (matmul k (matmul kf.observation_matrix kf.sqrt_covariance))
    ) in
    let updated_sqrt_cov = Tensor.(matmul i_kr kf.sqrt_covariance) in
    
    { kf with
      state = updated_state;
      sqrt_covariance = updated_sqrt_cov }

  (* Invariant update for improved numerical stability *)
  let update_invariant kf observation =
    let n = Tensor.shape kf.state |> List.hd in
    
    (* Pre-array construction *)
    let pre_array = Tensor.(cat [
      kf.sqrt_covariance;
      kf.sqrt_observation_noise;
      neg kf.observation_matrix
    ] ~dim:0) in
    
    (* QR decomposition *)
    let q, r = Tensor.qr pre_array in
    
    (* Extract components *)
    let r11 = Tensor.(narrow r ~dim:0 ~start:0 ~length:n) in
    let r12 = Tensor.(narrow r ~dim:0 ~start:n 
                               ~length:(Tensor.shape r |> List.hd)) in
    
    (* Compute Kalman gain *)
    let k = Tensor.(triangular_solve r11 r12 ~upper:true) in
    
    (* Update state *)
    let innovation = Tensor.(sub observation 
      (matmul kf.observation_matrix kf.state)) in
    let updated_state = Tensor.(add kf.state (matmul k innovation)) in
    
    (* New sqrt covariance is r11 *)
    { kf with
      state = updated_state;
      sqrt_covariance = r11 }
end

(* Particle Filter *)
module ParticleFilter = struct
  type particle = {
    state: state;
    weight: float;
  }

  type resampling_scheme =
    | Multinomial
    | Systematic
    | Stratified
    | Residual

  type t = {
    particles: particle array;
    n_particles: int;
    state_transition: state -> state;
    observation_likelihood: observation -> state -> float;
    resample_threshold: float;
    resampling_method: resampling_scheme;
  }

  (* Resampling methods *)
  let multinomial_resample particles n =
    let cumsum = Array.make n 0.0 in
    let _ = Array.fold_left (fun acc p ->
      let new_acc = acc +. p.weight in
      cumsum.(int_of_float (new_acc *. float_of_int n) - 1) <- new_acc;
      new_acc
    ) 0.0 particles in
    
    Array.init n (fun _ ->
      let u = Random.float 1.0 in
      let rec binary_search left right =
        if right < left then left
        else
          let mid = (left + right) / 2 in
          if cumsum.(mid) < u then
            binary_search (mid + 1) right
          else
            binary_search left (mid - 1)
      in
      let idx = binary_search 0 (Array.length particles - 1) in
      { particles.(idx) with weight = 1. /. float_of_int n }
    )

  let systematic_resample particles n =
    let n_float = float_of_int n in
    let u = Random.float (1. /. n_float) in
    let cumsum = Array.make n 0. in
    let _ = Array.fold_left (fun acc p ->
      let new_acc = acc +. p.weight in
      cumsum.(int_of_float (new_acc *. n_float) - 1) <- new_acc;
      new_acc
    ) 0. particles in
    
    Array.init n (fun i ->
      let u_i = u +. float_of_int i /. n_float in
      let idx = ref 0 in
      while !idx < n - 1 && u_i > cumsum.(!idx) do
        incr idx
      done;
      { particles.(!idx) with weight = 1. /. n_float }
    )

  let stratified_resample particles n =
    let n_float = float_of_int n in
    let cumsum = Array.make n 0. in
    let _ = Array.fold_left (fun acc p ->
      let new_acc = acc +. p.weight in
      cumsum.(int_of_float (new_acc *. n_float) - 1) <- new_acc;
      new_acc
    ) 0. particles in
    
    Array.init n (fun i ->
      let u = (Random.float 1.0 +. float_of_int i) /. n_float in
      let idx = ref 0 in
      while !idx < n - 1 && u > cumsum.(!idx) do
        incr idx
      done;
      { particles.(!idx) with weight = 1. /. n_float }
    )

  let residual_resample particles n =
    let n_float = float_of_int n in
    let deterministic_counts = Array.map (fun p -> 
      float_of_int (int_of_float (p.weight *. n_float))
    ) particles in
    
    let residual_weights = Array.map2 (fun p count ->
      (p.weight *. n_float -. count) /. 
      (n_float -. Array.fold_left (+.) 0.0 deterministic_counts)
    ) particles deterministic_counts in
    
    let deterministic_indices = Array.concat (
      Array.mapi (fun i count ->
        Array.make (int_of_float count) particles.(i)
      ) deterministic_counts
    ) in
    
    let residual_count = 
      n - Array.length deterministic_indices in
    
    if residual_count > 0 then
      Array.append deterministic_indices 
        (multinomial_resample 
           (Array.mapi (fun i p -> 
             {p with weight = residual_weights.(i)}) particles)
           residual_count)
    else
      deterministic_indices

  (* Create particle filter *)
  let create ~n_particles ~init_state ~state_trans ~obs_likelihood 
            ~resample_thresh ~resample_method = {
    particles = Array.init n_particles (fun _ -> {
      state = init_state;
      weight = 1. /. float_of_int n_particles;
    });
    n_particles;
    state_transition = state_trans;
    observation_likelihood = obs_likelihood;
    resample_threshold = resample_thresh;
    resampling_method = resample_method;
  }

  (* Calculate effective sample size *)
  let effective_sample_size pf =
    let sum_sq_weights = Array.fold_left (fun acc p ->
      acc +. (p.weight *. p.weight)
    ) 0. pf.particles in
    1. /. sum_sq_weights

  (* Predict and update step *)
  let predict_and_update pf observation =
    (* Predict step *)
    let predicted = Array.map (fun p -> {
      p with state = pf.state_transition p.state
    }) pf.particles in
    
    (* Update step *)
    let total_weight = ref 0. in
    Array.iter (fun p ->
      let likelihood = pf.observation_likelihood observation p.state in
      let new_weight = p.weight *. likelihood in
      p.weight <- new_weight;
      total_weight := !total_weight +. new_weight
    ) predicted;
    
    (* Normalize weights *)
    Array.iter (fun p ->
      p.weight <- p.weight /. !total_weight
    ) predicted;
    
    (* Resample if needed *)
    let ess = effective_sample_size pf in
    if ess < pf.resample_threshold then
      let resampled = match pf.resampling_method with
        | Multinomial -> multinomial_resample predicted pf.n_particles
        | Systematic -> systematic_resample predicted pf.n_particles
        | Stratified -> stratified_resample predicted pf.n_particles
        | Residual -> residual_resample predicted pf.n_particles in
      { pf with particles = resampled }
    else
      { pf with particles = predicted }

  (* Get state estimate *)
  let estimate pf =
    let state_sum = ref (Tensor.zeros_like pf.particles.(0).state) in
    Array.iter (fun p ->
      state_sum := Tensor.(add !state_sum (mul_scalar p.state p.weight))
    ) pf.particles;
    !state_sum
end

(* Black-Karasinski Model *)
module BlackKarasinskiModel = struct
  type term_structure = {
    times: float array;
    rates: float array;
    volatilities: float array;
  }

  type t = {
    params: parameters;
    term_structure: term_structure;
    dt: float;
  }

  let create ~mu ~theta ~sigma ~dt ~term_struct = {
    params = { mu; theta; sigma; rho = 0.; kappa = 0.; xi = 0. };
    term_structure = term_struct;
    dt
  }

  (* Short rate evolution *)
  let step_short_rate model rate =
    let drift = model.params.theta *. (model.params.mu -. log rate) in
    let diffusion = model.params.sigma *. sqrt model.dt in
    let noise = Utils.random_gaussian () in
    rate *. exp (drift *. model.dt +. diffusion *. noise)

  (* Forward rate calculation *)
  let forward_rate model t1 t2 =
    let r1 = Array.fold_left (fun acc i ->
      if model.term_structure.times.(i) <= t1 then
        model.term_structure.rates.(i)
      else acc) 0.0 model.term_structure.rates in
    let r2 = Array.fold_left (fun acc i ->
      if model.term_structure.times.(i) <= t2 then
        model.term_structure.rates.(i)
      else acc) 0.0 model.term_structure.rates in
    (r2 *. t2 -. r1 *. t1) /. (t2 -. t1)

  (* Full simulation *)
  let simulate model init_rate steps =
    let rec aux rate acc remaining =
      if remaining <= 0 then List.rev acc
      else
        let next_rate = step_short_rate model rate in
        aux next_rate (next_rate :: acc) (remaining - 1)
    in
    aux init_rate [init_rate] (steps - 1)

  (* Extended Kalman filter formulation *)
  let create_ekf model init_state =
    let state_transition state =
      let r = Tensor.get state [0] in
      let log_r = log r in
      let drift = model.params.theta *. (model.params.mu -. log_r) in
      Tensor.of_float1 [|r *. exp (drift *. model.dt)|] in
    
    let observation_fn state =
      state in
    
    let state_jacobian state =
      let r = Tensor.get state [0] in
      let derivative = 1. +. model.params.theta *. 
                      (1. -. model.params.mu /. log r) *. model.dt in
      Tensor.of_float2 [|[|derivative|]|] in
    
    let observation_jacobian _ =
      Tensor.eye 1 in
    
    let proc_noise = Tensor.of_float2
      [|[|model.params.sigma *. model.params.sigma *. model.dt|]|] in
    
    let obs_noise = Tensor.(mul_scalar (eye 1) 1e-6) in
    
    ExtendedKalmanFilter.create
      ~init_state
      ~init_cov:(Tensor.eye 1)
      ~proc_noise
      ~obs_noise
      ~state_trans:state_transition
      ~obs_func:observation_fn
      ~state_jac:state_jacobian
      ~obs_jac:observation_jacobian

  (* Complete MLE estimation *)  
  let mle_estimate data dt =
    let n = Array.length data - 1 in
    let log_likelihood params =
      let ll = ref 0. in
      for i = 0 to n - 1 do
        let r_curr = log data.(i+1) in
        let r_prev = log data.(i) in
        let mean = r_prev +. params.theta *. (params.mu -. r_prev) *. dt in
        let var = params.sigma *. params.sigma *. dt in
        ll := !ll +. (-0.5 *. (log (2. *. Float.pi *. var) +. 
                              (r_curr -. mean) ** 2. /. var))
      done;
      !ll in
    
    let rec optimize params iter =
      if iter >= 1000 then params
      else
        let eps = 1e-6 in
        let grad = {
          mu = (log_likelihood { params with mu = params.mu +. eps } -. 
                log_likelihood params) /. eps;
          theta = (log_likelihood { params with theta = params.theta +. eps } -. 
                  log_likelihood params) /. eps;
          sigma = (log_likelihood { params with sigma = params.sigma +. eps } -. 
                  log_likelihood params) /. eps;
          rho = 0.;  (* Not used in BK model *)
          kappa = 0.;
          xi = 0.;
        } in
        
        let lr = 0.01 in
        let new_params = {
          mu = params.mu +. lr *. grad.mu;
          theta = max 0. (params.theta +. lr *. grad.theta);
          sigma = max 0. (params.sigma +. lr *. grad.sigma);
          rho = 0.;
          kappa = 0.;
          xi = 0.;
        } in
        
        let improvement = log_likelihood new_params -. log_likelihood params in
        if abs_float improvement < 1e-6 then params
        else optimize new_params (iter + 1)
    in
    
    let init_params = {
      mu = log (Array.fold_left (+.) 0. data /. float_of_int (Array.length data));
      theta = 0.5;
      sigma = 0.2;
      rho = 0.;
      kappa = 0.;
      xi = 0.;
    } in
    
    optimize init_params 0
end

(* Jump Process *)
module JumpProcess = struct
  type jump_type =
    | PoissonJump of float * float  (* rate, mean size *)
    | CompoundPoisson of float * (unit -> float)  (* rate, size generator *)
    | VarianceGamma of float * float * float  (* variance rate, drift, volatility *)
    | NIG of float * float * float  (* tail heaviness, asymmetry, scale *)

  type levy_measure = {
    small_jumps: float -> float;  (* Lévy measure for small jumps *)
    large_jumps: float -> float;  (* Lévy measure for large jumps *)
    truncation: float;            (* Truncation level *)
  }

  type t = {
    base_process: [ `OU of OUProcess.t 
                 | `Heston of HestonModel.t
                 | `BK of BlackKarasinskiModel.t ];
    jump_type: jump_type;
    levy_measure: levy_measure;
    dt: float;
  }

  (* Jump simulation *)
  let simulate_jump t =
    match t.jump_type with
    | PoissonJump (rate, mean_size) ->
        if Random.float 1.0 < rate *. t.dt then
          mean_size +. Utils.random_gaussian () *. sqrt t.dt
        else 0.0
    | CompoundPoisson (rate, size_gen) ->
        if Random.float 1.0 < rate *. t.dt then
          size_gen ()
        else 0.0
    | VarianceGamma (var_rate, drift, vol) ->
        let g = Utils.random_gamma (t.dt /. var_rate) var_rate in
        drift *. g +. vol *. sqrt g *. Utils.random_gaussian ()
    | NIG (alpha, beta, delta) ->
        let ig = Utils.random_inverse_gaussian 
          (delta *. t.dt /. sqrt (alpha *. alpha -. beta *. beta))
          (delta *. t.dt) in
        beta *. ig +. sqrt ig *. Utils.random_gaussian ()

  (* Small jump approximation *)
  let approximate_small_jumps t =
    let n_steps = 1000 in
    let dx = t.levy_measure.truncation /. float_of_int n_steps in
    let sum = ref 0. in
    for i = 1 to n_steps do
      let x = float_of_int i *. dx in
      sum := !sum +. x *. x *. t.levy_measure.small_jumps x *. dx
    done;
    sqrt (!sum *. t.dt)

  (* Path simulation *)
  let simulate t init_state steps =
    let rec aux state acc remaining =
      if remaining <= 0 then List.rev acc
      else
        (* Simulate base process *)
        let base_step = match t.base_process with
        | `OU ou -> OUProcess.step ou state
        | `Heston h -> HestonModel.step_price h state (Tensor.ones [1])
        | `BK bk -> BlackKarasinskiModel.step_short_rate bk (Tensor.to_float0_exn state) in
        
        (* Add jumps *)
        let small_jumps = approximate_small_jumps t in
        let large_jump = simulate_jump t in
        
        let next_state = Tensor.(add_scalar (add_scalar base_step small_jumps) large_jump) in
        aux next_state (next_state :: acc) (remaining - 1)
    in
    aux init_state [init_state] (steps - 1)
end

(* Multivariate Jump Process *)
module MultivariateJumpProcess = struct
  type jump_correlation = {
    matrix: float array array;
    cholesky: float array array option;
  }

  type t = {
    components: JumpProcess.t array;
    correlation: jump_correlation;
    dt: float;
  }

  (* Compute Cholesky decomposition of correlation matrix *)
  let compute_cholesky matrix =
    let n = Array.length matrix in
    let l = Array.make_matrix n n 0. in
    
    for i = 0 to n - 1 do
      for j = 0 to i do
        let sum = ref 0. in
        if i = j then begin
          for k = 0 to j - 1 do
            sum := !sum +. l.(j).(k) *. l.(j).(k)
          done;
          l.(i).(j) <- sqrt (matrix.(i).(j) -. !sum)
        end else begin
          for k = 0 to j - 1 do
            sum := !sum +. l.(i).(k) *. l.(j).(k)
          done;
          l.(i).(j) <- (matrix.(i).(j) -. !sum) /. l.(j).(j)
        end
      done
    done;
    l

  (* Create multivariate jump process *)
  let create processes correlation_matrix dt = {
    components = processes;
    correlation = {
      matrix = correlation_matrix;
      cholesky = Some (compute_cholesky correlation_matrix);
    };
    dt;
  }

  (* Simulate correlated paths *)
  let simulate t init_states steps =
    let n = Array.length t.components in
    let paths = Array.make_matrix n steps (Tensor.to_float0_exn (Array.get init_states 0)) in
    
    (* Initialize first values *)
    Array.iteri (fun i v -> paths.(i).(0) <- Tensor.to_float0_exn v) init_states;
    
    (* Simulate paths *)
    for step = 1 to steps - 1 do
      (* Generate correlated standard normal variables *)
      let z = Array.init n (fun _ -> Utils.random_gaussian ()) in
      let corr_z = Array.init n (fun i ->
        let sum = ref 0. in
        for j = 0 to i do
          sum := !sum +. chol.(i).(j) *. z.(j)
        done;
        !sum
      ) in
      
      (* Simulate each component *)
      Array.iteri (fun i process ->
        let base_step = JumpProcess.simulate process 
          (Tensor.of_float1 [|paths.(i).(step-1)|]) 1 in
        let jump = JumpProcess.simulate_jump process in
        paths.(i).(step) <- 
          Tensor.to_float0_exn (List.hd base_step) +. 
          corr_z.(i) *. sqrt t.dt +. jump
      ) t.components
    done;
    paths
end

(* Parameter Estimation Framework *)
module ParameterEstimation = struct
  type estimation_method =
    | MaximumLikelihood
    | MethodOfMoments
    | GMM
    | MCMC of int  (* number of iterations *)

  type estimation_config = {
    method_type: estimation_method;
    learning_rate: float;
    tolerance: float;
    regularization: float;
    batch_size: int option;
  }

  (* Moment-based estimation *)
  module MomentEstimation = struct
    type moment_condition = {
      function_value: parameters -> float -> float;
      target_value: float;
      weight: float;
    }

    let estimate_moments conditions init_params data config =
      let objective params =
        let n = Array.length data in
        Array.fold_left (fun acc cond ->
          let empirical = Array.fold_left (fun sum x ->
            sum +. cond.function_value params x
          ) 0. data /. float_of_int n in
          acc +. cond.weight *. (empirical -. cond.target_value) ** 2.
        ) 0. conditions in
      
      let rec optimize params iter =
        if iter >= 1000 then params
        else
          (* Compute numerical gradients *)
          let eps = 1e-6 in
          let grad = {
            mu = (objective { params with mu = params.mu +. eps } -. 
                  objective params) /. eps;
            theta = (objective { params with theta = params.theta +. eps } -. 
                    objective params) /. eps;
            sigma = (objective { params with sigma = params.sigma +. eps } -. 
                    objective params) /. eps;
            rho = (objective { params with rho = params.rho +. eps } -. 
                  objective params) /. eps;
            kappa = (objective { params with kappa = params.kappa +. eps } -. 
                    objective params) /. eps;
            xi = (objective { params with xi = params.xi +. eps } -. 
                  objective params) /. eps;
          } in
          
          (* Update parameters with constraints *)
          let new_params = {
            mu = params.mu -. config.learning_rate *. grad.mu;
            theta = max 0. (params.theta -. config.learning_rate *. grad.theta);
            sigma = max 0. (params.sigma -. config.learning_rate *. grad.sigma);
            rho = max (-1.) (min 1. (params.rho -. config.learning_rate *. grad.rho));
            kappa = max 0. (params.kappa -. config.learning_rate *. grad.kappa);
            xi = max 0. (params.xi -. config.learning_rate *. grad.xi);
          } in
          
          let improvement = objective params -. objective new_params in
          if improvement < config.tolerance then params
          else optimize new_params (iter + 1) in
      
      optimize init_params 0
  end

  (* MCMC estimation *)
  module MCMCEstimation = struct
    type mcmc_config = {
      n_chains: int;
      burnin: int;
      thin: int;
      proposal_std: float;
    }

    let metropolis_hastings ~log_likelihood ~prior ~config data =
      let n_chains = config.n_chains in
      let n_iter = MCMC n
      
      (* Initialize chains *)
      let chains = Array.init n_chains (fun _ -> {
        mu = Random.float 1.0;
        theta = Random.float 1.0;
        sigma = Random.float 0.5;
        rho = Random.float_range (-0.8) 0.8;
        kappa = Random.float 2.0;
        xi = Random.float 0.5;
      }) in
      
      let chain_ll = Array.map (fun params ->
        log_likelihood params data +. prior params
      ) chains in
      
      (* Store samples after burnin and thinning *)
      let n_samples = (n_iter - config.burnin) / config.thin in
      let samples = Array.make_matrix n_chains n_samples chains.(0) in
      
      (* Run chains *)
      for iter = 0 to n_iter - 1 do
        Array.iteri (fun i current_params ->
          (* Propose new parameters *)
          let proposal = {
            mu = current_params.mu +. Utils.random_gaussian () *. config.proposal_std;
            theta = current_params.theta +. Utils.random_gaussian () *. config.proposal_std;
            sigma = abs_float (current_params.sigma +. 
                             Utils.random_gaussian () *. config.proposal_std);
            rho = max (-1.) (min 1. (current_params.rho +. 
                                    Utils.random_gaussian () *. config.proposal_std));
            kappa = abs_float (current_params.kappa +. 
                             Utils.random_gaussian () *. config.proposal_std);
            xi = abs_float (current_params.xi +. 
                          Utils.random_gaussian () *. config.proposal_std);
          } in
          
          (* Compute acceptance ratio *)
          let prop_ll = log_likelihood proposal data +. prior proposal in
          let ratio = exp (prop_ll -. chain_ll.(i)) in
          
          (* Accept/reject *)
          if Random.float 1.0 < ratio then begin
            chains.(i) <- proposal;
            chain_ll.(i) <- prop_ll
          end;
          
          (* Store sample *)
          if iter >= config.burnin && (iter - config.burnin) mod config.thin = 0 then
            samples.(i).((iter - config.burnin) / config.thin) <- chains.(i)
        ) chains
      done;
      
      samples
  end

  (* Generalized Method of Moments *)
  module GMMEstimation = struct
    let two_step_gmm moments instruments init_params data config =
      (* First step: identity weighting matrix *)
      let n = Array.length data in
      let n_moments = Array.length moments in
      let n_instruments = Array.length instruments in
      
      (* Compute moment conditions *)
      let compute_moments params =
        Array.init n_moments (fun i ->
          Array.init n_instruments (fun j ->
            let moment = moments.(i) in
            let instrument = instruments.(j) in
            Array.fold_left (fun acc x ->
              acc +. moment params x *. instrument x
            ) 0. data /. float_of_int n
          )
        ) in
      
      (* First step optimization *)
      let first_step = MomentEstimation.estimate_moments
        (Array.mapi (fun i m -> {
          MomentEstimation.function_value = m;
          target_value = 0.0;
          weight = 1.0;
        }) moments)
        init_params data config in
      
      (* Compute optimal weighting matrix *)
      let first_moments = compute_moments first_step in
      let s_matrix = Array.make_matrix n_moments n_moments 0. in
      for i = 0 to n_moments - 1 do
        for j = 0 to n_moments - 1 do
          s_matrix.(i).(j) <- 
            Array.fold_left2 (fun acc mi mj -> acc +. mi *. mj)
              0. first_moments.(i) first_moments.(j)
        done
      done;
      
      (* Second step with optimal weights *)
      let optimal_weights = Utils.safe_cholesky (Tensor.of_float2 s_matrix) in
      MomentEstimation.estimate_moments
        (Array.mapi (fun i m -> {
          MomentEstimation.function_value = m;
          target_value = 0.0;
          weight = Tensor.get optimal_weights [i; i];
        }) moments)
        first_step data config
  end
end

(* Numerical Methods *)
module NumericalMethods = struct
  (* Euler-Maruyama scheme *)
  let euler_maruyama ~drift ~diffusion ~init ~dt ~steps =
    let trajectory = Array.make steps init in
    for i = 1 to steps - 1 do
      let prev = trajectory.(i-1) in
      let drift_term = drift prev *. dt in
      let diff_term = diffusion prev *. sqrt dt *. Utils.random_gaussian () in
      trajectory.(i) <- prev +. drift_term +. diff_term
    done;
    trajectory

  (* Milstein scheme *)
  let milstein ~drift ~diffusion ~diffusion_derivative ~init ~dt ~steps =
    let trajectory = Array.make steps init in
    for i = 1 to steps - 1 do
      let prev = trajectory.(i-1) in
      let dw = Utils.random_gaussian () *. sqrt dt in
      let drift_term = drift prev *. dt in
      let diff_term = diffusion prev *. dw in
      let milstein_term = 
        0.5 *. diffusion prev *. diffusion_derivative prev *. (dw *. dw -. dt) in
      trajectory.(i) <- prev +. drift_term +. diff_term +. milstein_term
    done;
    trajectory

  (* Strong Order 1.5 Taylor scheme *)
  let taylor_1_5 ~drift ~diffusion ~drift_derivative ~diffusion_derivative 
                 ~mixed_derivative ~init ~dt ~steps =
    let trajectory = Array.make steps init in
    for i = 1 to steps - 1 do
      let prev = trajectory.(i-1) in
      let dw = Utils.random_gaussian () *. sqrt dt in
      let dz = Utils.random_gaussian () *. dt ** (1.5) in
      
      let drift_term = drift prev *. dt in
      let diff_term = diffusion prev *. dw in
      let drift_diff_term = 
        drift_derivative prev *. diffusion prev *. 
        (dw *. dt -. dz) in
      let diff_diff_term =
        0.5 *. diffusion prev *. diffusion_derivative prev *.
        (dw *. dw -. dt) in
      let mixed_term =
        mixed_derivative prev *. dz in
      
      trajectory.(i) <- prev +. drift_term +. diff_term +.
                       drift_diff_term +. diff_diff_term +. mixed_term
    done;
    trajectory

  (* Adaptive step size control *)
  let adaptive_step ~drift ~diffusion ~init ~dt ~tol ~max_steps =
    let rec integrate state time acc steps =
      if steps >= max_steps || time >= 1.0 then
        List.rev acc
      else
        let proposed_step = min dt (1.0 -. time) in
        
        (* Two step sizes for error estimation *)
        let step1 = state +. drift state *. proposed_step +.
                   diffusion state *. sqrt proposed_step *. 
                   Utils.random_gaussian () in
        
        let half_step = proposed_step /. 2. in
        let intermediate = state +. drift state *. half_step +.
                         diffusion state *. sqrt half_step *. 
                         Utils.random_gaussian () in
        let step2 = intermediate +. drift intermediate *. half_step +.
                   diffusion intermediate *. sqrt half_step *. 
                   Utils.random_gaussian () in
        
        (* Error estimation *)
        let error = abs_float (step2 -. step1) in
        
        if error < tol then
          (* Accept step *)
          integrate step2 (time +. proposed_step) 
                  (step2 :: acc) (steps + 1)
        else
          (* Reduce step size and retry *)
          let new_dt = proposed_step *. sqrt (tol /. error) in
          integrate state time acc steps ~dt:new_dt
    in
    integrate init 0.0 [init] 0
end