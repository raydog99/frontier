open Torch

module Complex = struct
  type t = {re: Tensor.t; im: Tensor.t}
  
  let create re im = {re; im}
  
  let zero () = create (Tensor.zeros []) (Tensor.zeros [])
  
  let add a b = create 
    (Tensor.add a.re b.re) 
    (Tensor.add a.im b.im)
    
  let mul a b = create
    (Tensor.sub 
      (Tensor.mm a.re b.re)
      (Tensor.mm a.im b.im))
    (Tensor.add
      (Tensor.mm a.re b.im)
      (Tensor.mm a.im b.re))
      
  let conj a = create a.re (Tensor.neg a.im)
  
  let abs_sq a = Tensor.add 
    (Tensor.mul a.re a.re) 
    (Tensor.mul a.im a.im)
    
  let scale a scalar = create
    (Tensor.mul_scalar a.re scalar)
    (Tensor.mul_scalar a.im scalar)
    
  let inverse a =
    let denom = abs_sq a in
    create
      (Tensor.div a.re denom)
      (Tensor.div (Tensor.neg a.im) denom)
end

module MatrixOps = struct
  type matrix_properties = {
    condition_number: float;
    rank: int;
    is_positive_definite: bool
  }
  
  let check_stability mat =
    let eigenvals = Tensor.symeig mat ~eigenvectors:false in
    let min_eig = Tensor.min eigenvals |> Tensor.to_float0_exn in
    let max_eig = Tensor.max eigenvals |> Tensor.to_float0_exn in
    {
      condition_number = max_eig /. min_eig;
      rank = Tensor.size mat 0; 
      is_positive_definite = min_eig > 0.
    }
    
  let robust_inverse mat =
    let props = check_stability mat in
    if props.condition_number > 1e6 then begin
      let u, s, v = Tensor.svd mat in
      let s_inv = Tensor.div_scalar s
        (Tensor.add s (Tensor.float_vec [1e-10])) in
      let v_t = Tensor.transpose v 0 1 in
      Tensor.mm (Tensor.mm v (Tensor.diag s_inv)) (Tensor.transpose u 0 1)
    end else
      Tensor.inverse mat
      
  let solve_symmetric_system a b =
    let props = check_stability a in
    let l = Tensor.cholesky a in
    let y = Tensor.triangular_solve b l ~upper:false ~transpose:false in
    Tensor.triangular_solve y l ~upper:false ~transpose:true
end

module SpectralEstimation = struct
  type spectral_matrix = {
    frequencies: Tensor.t;
    density: Complex.t array array;
    coherence: Tensor.t array array option;
    phase: Tensor.t array array option;
    error_bounds: Complex.t array array option;
  }
  
  let compute_autocovariance_matrix data max_lag =
    let t = Tensor.size data 0 in
    let n = Tensor.size data 1 in
    let result = Array.make_matrix n n (Tensor.zeros [max_lag + 1]) in
    
    for i = 0 to n-1 do
      let series_i = Tensor.select data ~dim:1 ~index:i in
      for j = i to n-1 do
        let series_j = Tensor.select data ~dim:1 ~index:j in
        let cov = Tensor.zeros [max_lag + 1] in
        
        for lag = 0 to max_lag do
          let x1 = Tensor.narrow series_i ~dim:0 ~start:0 ~length:(t-lag) in
          let x2 = Tensor.narrow series_j ~dim:0 ~start:lag ~length:(t-lag) in
          let c = Tensor.mean (Tensor.mul x1 x2) in
          Tensor.set cov lag c
        done;
        
        result.(i).(j) <- cov;
        if i <> j then
          result.(j).(i) <- cov
      done
    done;
    
    result

  let estimate_spectral_density_matrix data params =
    let t = Tensor.size data 0 in 
    let n = Tensor.size data 1 in
    let n_freq = t / 2 + 1 in
    
    let frequencies = 
      Tensor.linspace ~start:0. ~end_:Float.pi ~steps:n_freq Cpu in
      
    let density = Array.make_matrix n n
      (Complex.create (Tensor.zeros [n_freq]) (Tensor.zeros [n_freq])) in
      
    let acf = compute_autocovariance_matrix data (min t 50) in
    
    for i = 0 to n-1 do
      for j = i to n-1 do
        let gamma_ij = acf.(i).(j) in
        let spec_ij = Complex.create
          (Tensor.zeros [n_freq])
          (Tensor.zeros [n_freq]) in
          
        for k = 0 to Tensor.size gamma_ij 0 - 1 do
          let weight = 1. -. float k /. float (Tensor.size gamma_ij 0) in
          for f = 0 to n_freq-1 do
            let freq = Tensor.get frequencies f in
            let angle = Tensor.mul_scalar freq (float k) in
            let cos_term = Tensor.cos angle in
            let term = Tensor.mul_scalar
              (Tensor.get gamma_ij k)
              (weight *. Tensor.to_float0_exn cos_term) in
            density.(i).(j) <- Complex.add 
              density.(i).(j)
              (Complex.scale spec_ij (Tensor.to_float0_exn term))
          done
        done;
        
        if i <> j then
          density.(j).(i) <- Complex.conj density.(i).(j)
      done
    done;
    
    {
      frequencies;
      density;
      coherence = None;
      phase = None;
      error_bounds = None
    }
end

module ARMAModel = struct
  type model_params = {
    ar_coeffs: Tensor.t;
    ma_coeffs: Tensor.t;
    innovation_var: float;
  }

  type parameter_constraints = {
    ar_stationary: bool;
    ma_invertible: bool;
    max_ar_coef: float;
    max_ma_coef: float;
    min_variance: float;
  }

  let create ar_order ma_order =
    {
      ar_coeffs = Tensor.zeros [ar_order];
      ma_coeffs = Tensor.zeros [ma_order];
      innovation_var = 1.0
    }

  let check_stationarity ar_coeffs =
    let n = Tensor.size ar_coeffs 0 in
    if n = 0 then true
    else
      let companion = Tensor.zeros [n; n] in
      Tensor.copy_ 
        (Tensor.narrow companion ~dim:0 ~start:0 ~length:1)
        (Tensor.neg ar_coeffs);
      if n > 1 then
        for i = 1 to n-1 do
          Tensor.set companion i (i-1) 1.
        done;
      
      let eigenvals = Tensor.symeig companion ~eigenvectors:false in
      Tensor.max eigenvals |> Tensor.to_float0_exn < 1.0

  let check_invertibility ma_coeffs =
    let n = Tensor.size ma_coeffs 0 in
    if n = 0 then true
    else
      let companion = Tensor.zeros [n; n] in
      Tensor.copy_
        (Tensor.narrow companion ~dim:0 ~start:0 ~length:1)
        (Tensor.neg ma_coeffs);
      if n > 1 then
        for i = 1 to n-1 do
          Tensor.set companion i (i-1) 1.
        done;
      
      let eigenvals = Tensor.symeig companion ~eigenvectors:false in
      Tensor.max eigenvals |> Tensor.to_float0_exn < 1.0

  let enforce_constraints state constraints =
    let ar_coef_constrained = 
      if constraints.ar_stationary && not (check_stationarity state.ar_coeffs) then
        let scale = constraints.max_ar_coef /. 
          (Tensor.max state.ar_coeffs |> Tensor.to_float0_exn) in
        Tensor.mul_scalar state.ar_coeffs scale
      else state.ar_coeffs in
      
    let ma_coef_constrained =
      if constraints.ma_invertible && not (check_invertibility state.ma_coeffs) then
        let scale = constraints.max_ma_coef /.
          (Tensor.max state.ma_coeffs |> Tensor.to_float0_exn) in
        Tensor.mul_scalar state.ma_coeffs scale
      else state.ma_coeffs in
      
    let var_constrained =
      max state.innovation_var constraints.min_variance in
      
    {state with
     ar_coeffs = ar_coef_constrained;
     ma_coeffs = ma_coef_constrained;
     innovation_var = var_constrained}

  let spectral_density params freq =
    let ar_poly = Polynomial.create params.ar_coeffs in
    let ma_poly = Polynomial.create params.ma_coeffs in
    
    let z = Complex.create 
      (Tensor.cos freq)
      (Tensor.sin freq) in
      
    let ar_eval = Polynomial.evaluate ar_poly z in
    let ma_eval = Polynomial.evaluate ma_poly z in
    
    let numerator = Complex.mul ma_eval (Complex.conj ma_eval) in
    let denominator = Complex.mul ar_eval (Complex.conj ar_eval) in
    
    Complex.create
      (Tensor.mul_scalar 
         (Complex.abs_sq numerator)
         params.innovation_var)
      (Tensor.zeros [])

  let simulate params n =
    let p = Tensor.size params.ar_coeffs 0 in
    let q = Tensor.size params.ma_coeffs 0 in
    
    let innovations = Tensor.randn [n+max p q] in
    let x = Tensor.zeros [n+max p q] in
    
    for t = max p q to n+max p q - 1 do
      let ar_terms = ref (Tensor.zeros []) in
      for i = 0 to p-1 do
        ar_terms := Tensor.add !ar_terms
          (Tensor.mul 
             (Tensor.get params.ar_coeffs i)
             (Tensor.get x (t-i-1)))
      done;
      
      let ma_terms = ref (Tensor.zeros []) in
      for i = 0 to q-1 do
        ma_terms := Tensor.add !ma_terms
          (Tensor.mul
             (Tensor.get params.ma_coeffs i)
             (Tensor.get innovations (t-i-1)))
      done;
      
      Tensor.set x t 
        (Tensor.add !ar_terms !ma_terms |> Tensor.to_float0_exn)
    done;
    
    Tensor.narrow x ~dim:0 ~start:(max p q) ~length:n
end

module ParameterEstimation = struct
  type estimation_config = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    momentum: float;
    batch_size: int option;
  }

  let default_config = {
    max_iter = 1000;
    tolerance = 1e-6;
    learning_rate = 0.01;
    momentum = 0.9;
    batch_size = None;
  }

  let estimate_ar data p =
    let t = Tensor.size data 0 in
    let y = Tensor.narrow data ~dim:0 ~start:p ~length:(t-p) in
    let x = Tensor.zeros [t-p; p] in
    
    for i = 0 to p-1 do
      let x_i = Tensor.narrow data ~dim:0 ~start:(p-i-1) ~length:(t-p) in
      Tensor.copy_ (Tensor.select x ~dim:1 ~index:i) x_i
    done;
    
    let xtx = Tensor.mm (Tensor.transpose x 0 1) x in
    let xty = Tensor.mm (Tensor.transpose x 0 1) y in
    let coeffs = Tensor.mm (MatrixOps.robust_inverse xtx) xty in
    
    let residuals = Tensor.sub y (Tensor.mm x coeffs) in
    let var = Tensor.mean (Tensor.mul residuals residuals) in
    
    coeffs, Tensor.to_float0_exn var

  let estimate_ma data q =
    let t = Tensor.size data 0 in
    let innovations = Tensor.zeros [t] in
    let ma_coeffs = Tensor.zeros [q] in
    
    for i = q to t-1 do
      let pred = ref 0. in
      for j = 0 to q-1 do
        pred := !pred +.
          (Tensor.get ma_coeffs j |> Tensor.to_float0_exn) *.
          (Tensor.get innovations (i-j-1) |> Tensor.to_float0_exn)
      done;
      
      let innov = (Tensor.get data i |> Tensor.to_float0_exn) -. !pred in
      Tensor.set innovations i innov
    done;
    
    let var = Tensor.var innovations in
    ma_coeffs, Tensor.to_float0_exn var

  let optimize_parameters objective_fn initial_params config =
    let n_params = Tensor.size initial_params 0 in
    
    let m = ref (Tensor.zeros [n_params]) in
    let v = ref (Tensor.zeros [n_params]) in
    
    let rec iterate params iter best_value =
      if iter >= config.max_iter then
        params
      else
        let value, grad = objective_fn params in
        
        (* Update moment estimates *)
        m := Tensor.add
          (Tensor.mul_scalar !m config.momentum)
          (Tensor.mul_scalar grad (1. -. config.momentum));
          
        v := Tensor.add
          (Tensor.mul_scalar !v config.momentum)
          (Tensor.mul_scalar (Tensor.mul grad grad) (1. -. config.momentum));
          
        (* Update parameters *)
        let update = Tensor.div
          (Tensor.mul_scalar !m config.learning_rate)
          (Tensor.add
             (Tensor.sqrt !v)
             (Tensor.float_vec [1e-8])) in
             
        let new_params = Tensor.sub params update in
        
        if abs_float (value -. best_value) < config.tolerance then
          new_params
        else
          iterate new_params (iter + 1) (min value best_value) in
          
    iterate initial_params 0 Float.infinity

  let estimate_arma data p q config =
    let ar_init, _ = estimate_ar data p in
    let ma_init, _ = estimate_ma data q in
    
    let initial_params = {
      ARMAModel.ar_coeffs = ar_init;
      ma_coeffs = ma_init;
      innovation_var = 1.0
    } in
    
    let objective params =
      let spec = ARMAModel.spectral_density params in
      let data_spec = SpectralEstimation.estimate_spectral_density_matrix data in
      
      let loss = ref 0. in
      let grad = Tensor.zeros [p + q] in
      
      (* Compute spectral likelihood *)
      for j = 0 to Array.length data_spec.frequencies - 1 do
        let freq = Tensor.get data_spec.frequencies j in
        let model_spec = spec freq in
        let data_spec_j = data_spec.density.(0).(0) in
        
        loss := !loss +.
          (Complex.abs_sq (Complex.sub model_spec data_spec_j) |>
           Tensor.to_float0_exn)
      done;
      
      !loss, grad in
      
    optimize_parameters objective initial_params config
end

module KalmanFilter = struct
  type filter_state = {
    mean: Tensor.t;
    covariance: Tensor.t;
    gain: Tensor.t;
    innovation: Tensor.t;
    innovation_cov: Tensor.t;
    loglik: float;
  }

  type system_matrices = {
    transition: Tensor.t;
    observation: Tensor.t;
    system_noise: Tensor.t;
    observation_noise: Tensor.t;
  }

  let predict_step state system_matrices =
    let {mean; covariance; _} = state in
    let {transition; system_noise; _} = system_matrices in
    
    (* Predict state *)
    let pred_mean = Tensor.mm transition mean in
    
    (* Predict covariance *)
    let term1 = Tensor.mm transition covariance in
    let term2 = Tensor.mm term1 (Tensor.transpose transition 0 1) in
    let pred_cov = Tensor.add term2 system_noise in
    
    {state with 
     mean = pred_mean;
     covariance = pred_cov}

  let update_step state observation system_matrices =
    let {mean; covariance; _} = state in
    let {observation_matrix; observation_noise; _} = system_matrices in
    
    (* Compute innovation *)
    let predicted_obs = Tensor.mm observation_matrix mean in
    let innovation = Tensor.sub observation predicted_obs in
    
    (* Innovation covariance *)
    let term1 = Tensor.mm observation_matrix covariance in
    let term2 = Tensor.mm term1 (Tensor.transpose observation_matrix 0 1) in
    let innovation_cov = Tensor.add term2 observation_noise in
    
    (* Kalman gain *)
    let term3 = Tensor.mm covariance (Tensor.transpose observation_matrix 0 1) in
    let gain = Tensor.mm term3 (MatrixOps.robust_inverse innovation_cov) in
    
    (* Update state *)
    let new_mean = Tensor.add mean (Tensor.mm gain innovation) in
    let identity = Tensor.eye (Tensor.size covariance 0) in
    let temp = Tensor.sub identity (Tensor.mm gain observation_matrix) in
    let new_cov = Tensor.mm temp covariance in
    
    (* Compute log likelihood *)
    let n = Tensor.size innovation 0 in
    let logdet = Tensor.logdet innovation_cov |> Tensor.to_float0_exn in
    let quad = Tensor.mm 
      (Tensor.mm 
         (Tensor.transpose innovation 0 1)
         (MatrixOps.robust_inverse innovation_cov))
      innovation |>
    Tensor.to_float0_exn in
    let loglik = -0.5 *. (float n *. log (2. *. Float.pi) +. 
                         logdet +. quad) in
    
    {state with
     mean = new_mean;
     covariance = new_cov;
     gain;
     innovation;
     innovation_cov;
     loglik}

  let smooth_state forward_states system_matrices =
    let n = Array.length forward_states in
    let smoothed_states = Array.copy forward_states in
    
    for t = n - 2 downto 0 do
      let state_t = forward_states.(t) in
      let state_t1 = smoothed_states.(t + 1) in
      
      (* Compute smoothing gain *)
      let pred_cov = Tensor.mm
        (Tensor.mm state_t.covariance 
           (Tensor.transpose system_matrices.transition 0 1))
        (MatrixOps.robust_inverse state_t1.covariance) in
        
      (* Update mean and covariance *)
      let innovation = Tensor.sub 
        state_t1.mean
        (Tensor.mm system_matrices.transition state_t.mean) in
        
      let new_mean = Tensor.add
        state_t.mean
        (Tensor.mm pred_cov innovation) in
        
      let new_cov = Tensor.add
        state_t.covariance
        (Tensor.mm
           (Tensor.mm pred_cov 
              (Tensor.sub 
                 state_t1.covariance
                 state_t1.covariance))
           (Tensor.transpose pred_cov 0 1)) in
           
      smoothed_states.(t) <- {state_t with
                             mean = new_mean;
                             covariance = new_cov}
    done;
    
    smoothed_states
end

module EMAlgorithm = struct
  type em_state = {
    params: ARMAModel.model_params;
    filtered_states: KalmanFilter.filter_state array;
    smoothed_states: KalmanFilter.filter_state array;
    loglik: float;
    iteration: int;
    converged: bool;
  }

  let create_initial_state data =
    let t = Tensor.size data 0 in
    let n = Tensor.size data 1 in
    
    (* Initialize with reasonable starting values *)
    let ar_coeffs = Tensor.ones [2] in  (* AR(1) parameters *)
    let ma_coeffs = Tensor.ones [2] in  (* MA(1) parameters *)
    
    let params = {
      ARMAModel.ar_coeffs;
      ma_coeffs;
      innovation_var = 1.0
    } in
    
    let initial_state = {
      KalmanFilter.mean = Tensor.zeros [n];
      covariance = Tensor.eye n;
      gain = Tensor.zeros [n; n];
      innovation = Tensor.zeros [n];
      innovation_cov = Tensor.zeros [n; n];
      loglik = Float.neg_infinity;
    } in
    
    {
      params;
      filtered_states = Array.make t initial_state;
      smoothed_states = Array.make t initial_state;
      loglik = Float.neg_infinity;
      iteration = 0;
      converged = false;
    }

  let e_step state data system_matrices =
    let t = Tensor.size data 0 in
    let filtered_states = Array.make t state.filtered_states.(0) in
    
    (* Forward pass *)
    let current_state = ref state.filtered_states.(0) in
    for i = 0 to t-1 do
      let obs = Tensor.select data ~dim:0 ~index:i in
      
      (* Predict *)
      current_state := 
        KalmanFilter.predict_step !current_state system_matrices;
        
      (* Update *)
      current_state :=
        KalmanFilter.update_step !current_state obs system_matrices;
        
      filtered_states.(i) <- !current_state
    done;
    
    (* Backward pass *)
    let smoothed_states = 
      KalmanFilter.smooth_state filtered_states system_matrices in
      
    {state with
     filtered_states;
     smoothed_states;
     loglik = Array.fold_left (fun acc s -> acc +. s.loglik)
       0. filtered_states}

  let m_step state data =
    let t = Tensor.size data 0 in
    
    (* Update AR parameters *)
    let ar_terms = ref (Tensor.zeros [t - 1; 2]) in
    let ar_targets = ref (Tensor.zeros [t - 1]) in
    
    for i = 1 to t-1 do
      let state_i = state.smoothed_states.(i).mean in
      let state_im1 = state.smoothed_states.(i-1).mean in
      
      Tensor.copy_
        (Tensor.narrow !ar_terms ~dim:0 ~start:(i-1) ~length:1)
        state_im1;
        
      Tensor.copy_
        (Tensor.narrow !ar_targets ~dim:0 ~start:(i-1) ~length:1)
        state_i
    done;
    
    let new_ar_coeffs = ParameterEstimation.estimate_ar !ar_targets 2 |> fst in
    
    (* Update MA parameters *)
    let ma_terms = ref (Tensor.zeros [t - 1; 2]) in
    let ma_targets = ref (Tensor.zeros [t - 1]) in
    
    for i = 1 to t-1 do
      let innov_i = state.filtered_states.(i).innovation in
      let innov_im1 = state.filtered_states.(i-1).innovation in
      
      Tensor.copy_
        (Tensor.narrow !ma_terms ~dim:0 ~start:(i-1) ~length:1)
        innov_im1;
        
      Tensor.copy_
        (Tensor.narrow !ma_targets ~dim:0 ~start:(i-1) ~length:1)
        innov_i
    done;
    
    let new_ma_coeffs = ParameterEstimation.estimate_ma !ma_targets 2 |> fst in
    
    (* Update innovation variance *)
    let total_var = ref 0. in
    for i = 0 to t-1 do
      total_var := !total_var +.
        (Tensor.get state.filtered_states.(i).innovation_cov 0 0 |>
         Tensor.to_float0_exn)
    done;
    let new_var = !total_var /. float t in
    
    {state with
     params = {
       ar_coeffs = new_ar_coeffs;
       ma_coeffs = new_ma_coeffs;
       innovation_var = new_var
     }}

  let check_convergence prev_ll curr_ll epsilon =
    abs_float (curr_ll -. prev_ll) < epsilon

  let run ?(max_iter=100) ?(epsilon=1e-6) data =
    let initial_state = create_initial_state data in
    
    let rec iterate state =
      if state.iteration >= max_iter || state.converged then
        state
      else
        let prev_ll = state.loglik in
        
        (* E-step *)
        let system_matrices = {
          KalmanFilter.transition = (* Compute from ARMA params *) 
            Tensor.zeros [1];
          observation = (* Compute from ARMA params *)
            Tensor.zeros [1];
          system_noise = Tensor.eye 1;
          observation_noise = 
            Tensor.mul_scalar (Tensor.eye 1) state.params.innovation_var;
        } in
        
        let state' = e_step state data system_matrices in
        
        (* M-step *)
        let state'' = m_step state' data in
        
        let converged = 
          check_convergence prev_ll state''.loglik epsilon in
          
        iterate {
          state'' with
          iteration = state''.iteration + 1;
          converged
        } in
        
    iterate initial_state
end