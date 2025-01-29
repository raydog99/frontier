open Torch

type term_structure = {
  term: float;
  yield: float;
}

type yield_curve = {
  date: float;  (* Unix timestamp *)
  points: term_structure array;
}

type nelson_siegel_params = {
  beta1: float;  (* Level *)
  beta2: float;  (* Slope *)
  beta3: float;  (* Curvature *)
  lambda: float; (* Decay rate *)
}

type integration_method = 
  | Trapezoidal
  | Simpson
  | RectangularLeft
  | RectangularRight

type basis_type =
  | FourierBasis of {
      max_freq: int;
      period: float;
    }
  | NelsonSiegelBasis
  | ExponentialBasis of {
      n_terms: int;
      rate: float array;
    }
  | GaussianBasis of {
      n_terms: int;
      centers: float array;
      width: float;
    }

let create_from_float_array arr =
    Tensor.of_float_array1 arr [Array.length arr]

let tensor_to_float_array t =
  Tensor.to_float1 t |> Array.of_list

let zeros shape = Tensor.zeros shape ~kind:Float

let ones shape = Tensor.ones shape ~kind:Float

let randn shape ~mean ~std =
  Tensor.randn shape ~mean ~std ~kind:Float

let nelson_siegel_basis tau lambda =
  let open Float in
  let exp_term = exp (-lambda *. tau) in
  let term1 = 1.0 in
  let term2 = (1.0 -. exp_term) /. (lambda *. tau) in
  let term3 = term2 -. exp_term in
  [|term1; term2; term3|]

let nelson_siegel_yield params tau =
  let {beta1; beta2; beta3; lambda} = params in
  let basis = nelson_siegel_basis tau lambda in
  beta1 *. basis.(0) +. 
  beta2 *. basis.(1) +. 
  beta3 *. basis.(2)

let estimate_nelson_siegel_params points =
  let open Torch in
  let n = Array.length points in
  
  (* Initialize parameters as trainable tensors *)
  let beta1 = Var_store.new_var ~name:"beta1" ~shape:[1] () in
  let beta2 = Var_store.new_var ~name:"beta2" ~shape:[1] () in
  let beta3 = Var_store.new_var ~name:"beta3" ~shape:[1] () in
  let lambda = 0.1 in
  
  (* Create optimization target *)
  let opt = Optimizer.adam [beta1; beta2; beta3] ~learning_rate:0.01 in
  
  (* Convert data to tensors *)
  let terms = Array.map (fun p -> p.term) points in
  let yields = Array.map (fun p -> p.yield) points in
  let x = Tensor.of_float_array1 terms [n] in
  let y = Tensor.of_float_array1 yields [n] in
  
  (* Training loop *)
  for _ = 1 to 1000 do
    Optimizer.zero_grad opt;
    
    (* Forward pass - compute Nelson-Siegel yields *)
    let basis = Tensor.stack 
      [Tensor.ones [n];
       Tensor.((sub (ones [n]) (exp (mul_scalar x (-.lambda)))) / 
         (mul_scalar x lambda));
       Tensor.((sub (sub (ones [n]) (exp (mul_scalar x (-.lambda)))) 
                   (exp (mul_scalar x (-.lambda)))) / 
         (mul_scalar x lambda))] 0 in
    
    let pred = Tensor.(
      add (mul (reshape beta1 [1;1]) (select basis 0))
        (add (mul (reshape beta2 [1;1]) (select basis 1))
             (mul (reshape beta3 [1;1]) (select basis 2)))
    ) in
    
    (* Compute MSE loss *)
    let loss = Tensor.(mean (pow (sub pred y) (Float 2.))) in
    
    (* Backward pass *)
    backward loss;
    Optimizer.step opt;
  done;
  
  (* Extract optimized parameters *)
  {
    beta1 = Tensor.to_float0 beta1;
    beta2 = Tensor.to_float0 beta2;
    beta3 = Tensor.to_float0 beta3;
    lambda;
  }

(* Robust matrix operations *)
module RobustMatrix = struct
  (* Condition number estimation *)
  let estimate_condition_number mat =
    let s = Tensor.svd mat |> fun (_, s, _) -> s in
    let max_s = Tensor.max s |> Tensor.to_float0 in
    let min_s = Tensor.min s |> Tensor.to_float0 in
    if min_s <= 0. then infinity
    else max_s /. min_s

  (* Modified Gram-Schmidt orthogonalization *)
  let modified_gram_schmidt mat =
    let m, n = Tensor.shape2_exn mat in
    let q = Tensor.clone mat in
    let r = Tensor.zeros [n; n] ~kind:Float in
    
    for k = 0 to n-1 do
      (* Compute column norm *)
      let col_k = Tensor.select q 1 k in
      let r_kk = Tensor.norm col_k in
      Tensor.set r [k; k] r_kk;
      
      (* Normalize column *)
      let q_k = Tensor.div_scalar col_k r_kk in
      Tensor.copy_ q_k (Tensor.select q 1 k);
      
      (* Orthogonalize remaining columns *)
      for j = k+1 to n-1 do
        let col_j = Tensor.select q 1 j in
        let r_kj = Tensor.dot q_k col_j in
        Tensor.set r [k; j] r_kj;
        
        let update = Tensor.mul_scalar q_k r_kj in
        Tensor.sub_ col_j update
      done
    done;
    
    q, r

  (* Robust Cholesky with pivoting *)
  let robust_cholesky mat epsilon =
    let n = Tensor.shape mat |> List.hd in
    let p = Tensor.eye n ~kind:Float in
    let l = Tensor.zeros [n; n] ~kind:Float in
    
    for k = 0 to n-1 do
      (* Find pivot *)
      let mut_max = ref neg_infinity in
      let mut_pivot = ref k in
      
      for i = k to n-1 do
        let val_ = Tensor.get mat [i; i] |> Tensor.to_float0 in
        if val_ > !mut_max then (
          mut_max := val_;
          mut_pivot := i
        )
      done;
      
      (* Swap if necessary *)
      if !mut_pivot <> k then (
        for j = 0 to n-1 do
          let tmp = Tensor.get mat [k; j] |> Tensor.to_float0 in
          Tensor.set mat [k; j] 
            (Tensor.get mat [!mut_pivot; j] |> Tensor.to_float0);
          Tensor.set mat [!mut_pivot; j] tmp;
          
          let tmp_p = Tensor.get p [k; j] |> Tensor.to_float0 in
          Tensor.set p [k; j] 
            (Tensor.get p [!mut_pivot; j] |> Tensor.to_float0);
          Tensor.set p [!mut_pivot; j] tmp_p
        done
      );
      
      (* Compute factorization *)
      let a_kk = Tensor.get mat [k; k] |> Tensor.to_float0 in
      if a_kk < epsilon then
        raise (ModelError (NumericalInstability 
          "Matrix not positive definite"));
      
      let l_kk = sqrt a_kk in
      Tensor.set l [k; k] l_kk;
      
      for i = k+1 to n-1 do
        let a_ik = Tensor.get mat [i; k] |> Tensor.to_float0 in
        Tensor.set l [i; k] (a_ik /. l_kk)
      done;
      
      for j = k+1 to n-1 do
        for i = j to n-1 do
          let a_ij = Tensor.get mat [i; j] |> Tensor.to_float0 in
          let l_ik = Tensor.get l [i; k] |> Tensor.to_float0 in
          let l_jk = Tensor.get l [j; k] |> Tensor.to_float0 in
          Tensor.set mat [i; j] (a_ij -. l_ik *. l_jk)
        done
      done
    done;
    
    l, p
end

(* Optimization methods *)
module Optimization = struct
  type optimizer_config = {
    method_type: [`Adam | `LBFGS | `SGD];
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    momentum: float option;
    beta1: float;  (* Adam parameter *)
    beta2: float;  (* Adam parameter *)
  }

  let default_optimizer_config = {
    method_type = `Adam;
    max_iter = 1000;
    tolerance = 1e-6;
    learning_rate = 1e-3;
    momentum = Some 0.9;
    beta1 = 0.9;
    beta2 = 0.999;
  }

  (* Adam optimizer *)
  let adam ~f ~init ~config =
    let m = ref (Tensor.zeros_like init) in
    let v = ref (Tensor.zeros_like init) in
    let beta1_t = ref 1.0 in
    let beta2_t = ref 1.0 in
    
    let rec optimize iter x =
      if iter >= config.max_iter then x
      else
        let loss, grad = f x in
        
        (* Update biased first moment estimate *)
        m := Tensor.add 
          (Tensor.mul_scalar !m config.beta1)
          (Tensor.mul_scalar grad (1.0 -. config.beta1));
        
        (* Update biased second moment estimate *)
        v := Tensor.add
          (Tensor.mul_scalar !v config.beta2)
          (Tensor.mul_scalar 
             (Tensor.pow grad (Tensor.float_vec [2.0])) 
             (1.0 -. config.beta2));
        
        (* Update bias correction terms *)
        beta1_t := !beta1_t *. config.beta1;
        beta2_t := !beta2_t *. config.beta2;
        
        (* Compute bias-corrected moment estimates *)
        let m_hat = Tensor.div_scalar !m (1.0 -. !beta1_t) in
        let v_hat = Tensor.div_scalar !v (1.0 -. !beta2_t) in
        
        (* Update parameters *)
        let update = Tensor.div m_hat
          (Tensor.add (Tensor.sqrt v_hat)
             (Tensor.full_like v_hat config.tolerance)) in
        let next_x = Tensor.sub x 
          (Tensor.mul_scalar update config.learning_rate) in
        
        optimize (iter + 1) next_x
    in
    optimize 0 init

  (* L-BFGS implementation *)
  let lbfgs ~f ~init ~config =
    let n = Tensor.shape init |> List.hd in
    let m = 10 in  (* Memory size *)
    
    (* Initialize storage for position and gradient differences *)
    let s = Array.make m (Tensor.zeros [n] ~kind:Float) in
    let y = Array.make m (Tensor.zeros [n] ~kind:Float) in
    let rho = Array.make m 0. in
    
    let rec optimize iter x =
      if iter >= config.max_iter then x
      else
        let loss, grad = f x in
        
        if iter > 0 then (
          (* Update position and gradient differences *)
          let idx = (iter - 1) mod m in
          s.(idx) <- Tensor.sub x !prev_x;
          y.(idx) <- Tensor.sub grad !prev_grad;
          rho.(idx) <- 1. /. (Tensor.dot y.(idx) s.(idx) |> 
                            Tensor.to_float0)
        );
        
        (* Two-loop recursion *)
        let q = Tensor.clone grad in
        let alpha = Array.make m 0. in
        
        (* First loop *)
        for i = min (iter - 1) (m - 1) downto 0 do
          let idx = i mod m in
          alpha.(i) <- rho.(idx) *. (Tensor.dot s.(idx) q |> 
                                   Tensor.to_float0);
          Tensor.sub_ q (Tensor.mul_scalar y.(idx) alpha.(i))
        done;
        
        (* Scale initial Hessian approximation *)
        if iter > 0 then (
          let idx = (iter - 1) mod m in
          let scale = (Tensor.dot s.(idx) y.(idx) |> Tensor.to_float0) /.
                     (Tensor.dot y.(idx) y.(idx) |> Tensor.to_float0) in
          Tensor.mul_scalar_ q scale
        );
        
        (* Second loop *)
        for i = 0 to min (iter - 1) (m - 1) do
          let idx = i mod m in
          let beta = rho.(idx) *. (Tensor.dot y.(idx) q |> 
                                 Tensor.to_float0) in
          Tensor.add_ q (Tensor.mul_scalar s.(idx) (alpha.(i) -. beta))
        done;
        
        (* Update parameters *)
        let direction = Tensor.neg q in
        let step_size = config.learning_rate in
        let next_x = Tensor.add x 
          (Tensor.mul_scalar direction step_size) in
        
        prev_x := x;
        prev_grad := grad;
        
        optimize (iter + 1) next_x
    in
    let prev_x = ref init in
    let prev_grad = ref (Tensor.zeros [n] ~kind:Float) in
    
    optimize 0 init
end

(* D-Operator implementations *)
let derivative f x h =
  (f (x +. h) -. f (x -. h)) /. (2. *. h)

let second_derivative f x h =
  (f (x +. h) -. 2. *. f x +. f (x -. h)) /. (h *. h)

let d_operator_matrix points h =
  let n = Array.length points in
  let mat = Tensor.zeros [n; n] ~kind:Float in
  
  (* Central difference for interior points *)
  for i = 1 to n-2 do
    Tensor.set mat [i; i-1] (-1.0);
    Tensor.set mat [i; i+1] 1.0
  done;
  
  (* Forward difference for first point *)
  Tensor.set mat [0; 0] (-1.0);
  Tensor.set mat [0; 1] 1.0;
  
  (* Backward difference for last point *)
  Tensor.set mat [n-1; n-2] (-1.0);
  Tensor.set mat [n-1; n-1] 1.0;
  
  Tensor.div_scalar mat (2. *. h)

let d2_operator_matrix points h =
  let n = Array.length points in
  let mat = Tensor.zeros [n; n] ~kind:Float in
  
  (* Central difference for interior points *)
  for i = 1 to n-2 do
    Tensor.set mat [i; i-1] 1.0;
    Tensor.set mat [i; i] (-2.0);
    Tensor.set mat [i; i+1] 1.0
  done;
    
(* Gaussian Process *)
module GP = struct
  type kernel_type =
    | RBF
    | Linear
    | Periodic of float  (* period *)
    | Composite of kernel_type list * [`Sum | `Product]

  (* Enhanced kernel parameters with time-varying components *)
  type kernel_params = {
    base_length_scale: float;
    base_signal_variance: float;
    noise_variance: float;
    time_decay: float option;        (* For time-varying length scale *)
    amplitude_growth: float option;  (* For time-varying signal variance *)
  }

  type gp_model = {
    params: kernel_params;
    kernel_type: kernel_type;
    mean: Tensor.t option;
  }

  (* Time-varying parameter handling *)
  let get_length_scale params t =
    match params.time_decay with
    | None -> params.base_length_scale
    | Some decay -> 
        params.base_length_scale *. Float.exp (-. decay *. t)

  let get_signal_variance params t =
    match params.amplitude_growth with
    | None -> params.base_signal_variance
    | Some growth ->
        params.base_signal_variance *. (1. +. growth *. t)

  (* Efficient kernel computation using broadcasting *)
  let compute_kernel_matrix x1 x2 params kernel_type t =
    let n1 = Tensor.shape x1 |> List.hd in
    let n2 = Tensor.shape x2 |> List.hd in
    
    let expanded_x1 = Tensor.unsqueeze x1 ~dim:1 in
    let expanded_x2 = Tensor.unsqueeze x2 ~dim:0 in
    
    let length_scale = get_length_scale params t in
    let signal_var = get_signal_variance params t in
    
    match kernel_type with
    | RBF ->
        let diff = Tensor.sub expanded_x1 expanded_x2 in
        let sq_dist = Tensor.pow diff (Tensor.float_vec [2.0]) in
        let scaled_dist = Tensor.div_scalar sq_dist 
          (2. *. length_scale *. length_scale) in
        Tensor.mul_scalar (Tensor.exp (Tensor.neg scaled_dist)) signal_var
        
    | Linear ->
        Tensor.mul_scalar (Tensor.mm x1 (Tensor.transpose2 x2)) signal_var
        
    | Periodic period ->
        let diff = Tensor.sub expanded_x1 expanded_x2 in
        let scaled_diff = Tensor.div_scalar diff period in
        let sin_term = Tensor.sin 
          (Tensor.mul_scalar scaled_diff Float.pi) in
        let sq_sin = Tensor.pow sin_term (Tensor.float_vec [2.0]) in
        let scaled_dist = Tensor.div_scalar sq_sin 
          (length_scale *. length_scale) in
        Tensor.mul_scalar (Tensor.exp (Tensor.neg scaled_dist)) signal_var
        
    | Composite (kernels, op) ->
        let compute_single k = compute_kernel_matrix x1 x2 params k t in
        let matrices = List.map compute_single kernels in
        match op with
        | `Sum -> List.fold_left Tensor.add (List.hd matrices) 
            (List.tl matrices)
        | `Product -> List.fold_left Tensor.mul (List.hd matrices) 
            (List.tl matrices)

  (* Create GP model *)
  let create_model ?(mean=None) params =
    {
      params;
      kernel_type = RBF;  (* Default to RBF kernel *)
      mean;
    }

  (* GP prediction with uncertainty *)
  let predict model x_train y_train x_test t =
    validate_matrix_properties x_train "training inputs";
    validate_matrix_properties y_train "training outputs";
    validate_matrix_properties x_test "test inputs";
    
    let k_train_train = compute_kernel_matrix x_train x_train 
      model.params model.kernel_type t in
    let k_test_train = compute_kernel_matrix x_test x_train 
      model.params model.kernel_type t in
    let k_test_test = compute_kernel_matrix x_test x_test 
      model.params model.kernel_type t in
    
    (* Add noise to training kernel *)
    let noise_diag = Tensor.eye (Tensor.shape k_train_train |> List.hd) 
      ~kind:Float in
    Tensor.mul_scalar_ noise_diag model.params.noise_variance;
    Tensor.add_ k_train_train noise_diag;
    
    (* Validate kernel matrices *)
    validate_positive_definite k_train_train "training kernel";
    
    (* Compute predictive distribution *)
    let l = Tensor.cholesky k_train_train in
    let alpha = Tensor.triangular_solve y_train ~upper:false ~a:l |> fst in
    let alpha = Tensor.triangular_solve alpha ~upper:true 
      ~a:(Tensor.transpose2 l) |> fst in
    
    (* Mean prediction *)
    let pred_mean = Tensor.mm k_test_train alpha in
    
    (* Add prior mean if specified *)
    let pred_mean = match model.mean with
      | Some prior_mean -> Tensor.add pred_mean prior_mean
      | None -> pred_mean in
    
    (* Variance prediction *)
    let v = Tensor.triangular_solve k_test_train ~upper:false ~a:l |> fst in
    let pred_var = Tensor.sub k_test_test 
      (Tensor.mm v (Tensor.transpose2 v)) in
    
    pred_mean, pred_var

  (* Marginal likelihood computation *)
  let compute_marginal_likelihood model x y =
    let n = Tensor.shape x |> List.hd in
    
    (* Compute kernel matrix *)
    let k = compute_kernel_matrix x x model.params model.kernel_type 0.0 in
    
    (* Add noise *)
    let noise_diag = Tensor.eye n ~kind:Float in
    Tensor.mul_scalar_ noise_diag model.params.noise_variance;
    Tensor.add_ k noise_diag;
    
    (* Compute log likelihood *)
    let l = Tensor.cholesky k in
    let alpha = Tensor.triangular_solve y ~upper:false ~a:l |> fst in
    let alpha = Tensor.triangular_solve alpha ~upper:true 
      ~a:(Tensor.transpose2 l) |> fst in
    
    let term1 = Tensor.dot y alpha |> Tensor.to_float0 in
    let term2 = 2. *. (Tensor.sum (Tensor.log (Tensor.diag l)) |> 
      Tensor.to_float0) in
    let term3 = float_of_int n *. Float.log (2. *. Float.pi) in
    
    -0.5 *. (term1 +. term2 +. term3)

(* Time Series *)
module TimeSeries = struct
  type var_model = {
    coefficients: Tensor.t array;  (* Array of coefficient matrices for each lag *)
    intercept: Tensor.t;          (* Intercept term *)
  }

  (* Fit VAR model *)
  let fit_var_model data n_lags =
    let n_samples = Array.length data in
    let n_features = Array.length data.(0).points in
    
    (* Create design matrix X and target matrix Y *)
    let n_effective = n_samples - n_lags in
    let x = Tensor.empty [n_effective; n_lags * n_features] ~kind:Float in
    let y = Tensor.empty [n_effective; n_features] ~kind:Float in
    
    (* Fill matrices *)
    for i = 0 to n_effective - 1 do
      (* Fill Y *)
      for j = 0 to n_features - 1 do
        Tensor.set y [i; j] data.(i + n_lags).points.(j).yield
      done;
      
      (* Fill X *)
      for lag = 0 to n_lags - 1 do
        for j = 0 to n_features - 1 do
          Tensor.set x [i; lag * n_features + j] 
            data.(i + n_lags - lag - 1).points.(j).yield
        done
      done
    done;
    
    (* Add constant term *)
    let x = Tensor.cat [Tensor.ones [n_effective; 1] ~kind:Float; x] 1 in
    
    (* Solve using OLS: β = (X'X)^(-1)X'y *)
    let xtx = Tensor.mm (Tensor.transpose2 x) x in
    validate_positive_definite xtx "VAR design matrix";
    
    let xty = Tensor.mm (Tensor.transpose2 x) y in
    let beta = Tensor.mm (Tensor.inverse xtx) xty in
    
    (* Extract coefficients and intercept *)
    let intercept = Tensor.narrow beta 0 0 1 in
    let coef = Array.init n_lags (fun i ->
      Tensor.narrow beta 0 (1 + i * n_features) n_features) in
    
    { coefficients = coef; intercept }

  (* Forecast using VAR model *)
  let forecast model data =
    let n_features = Array.length data.(0).points in
    let n_lags = Array.length model.coefficients in
    
    (* Prepare input *)
    let x = Tensor.empty [1; n_lags * n_features] ~kind:Float in
    
    (* Fill latest observations *)
    for lag = 0 to n_lags - 1 do
      for j = 0 to n_features - 1 do
        Tensor.set x [0; lag * n_features + j] 
          data.(Array.length data - lag - 1).points.(j).yield
      done
    done;
    
    (* Add constant term *)
    let x = Tensor.cat [Tensor.ones [1; 1] ~kind:Float; x] 1 in
    
    (* Compute forecast *)
    let pred = Tensor.mm x (Tensor.cat 
      (Array.to_list (Array.append [|model.intercept|] model.coefficients)) 1) in
    
    pred

  (* Information criteria computation *)
  let compute_information_criteria model data n_params =
    let predictions = forecast model data in
    let actuals = Tensor.of_float_array2 
      (Array.map (fun d -> Array.map (fun p -> p.yield) d.points) data) in
    
    let residuals = Tensor.sub actuals predictions in
    let n_obs = Tensor.shape residuals |> List.hd in
    
    (* Log likelihood computation *)
    let sigma = Tensor.mm (Tensor.transpose2 residuals) residuals in
    let sigma = Tensor.div_scalar sigma (float_of_int n_obs) in
    let log_det_sigma = Tensor.logdet sigma in
    
    let ll = float_of_int (-n_obs / 2) *. 
      (log_det_sigma +. Float.log (2. *. Float.pi) +. 1.) in
    
    (* Compute AIC and BIC *)
    let aic = -2. *. ll +. 2. *. float_of_int n_params in
    let bic = -2. *. ll +. float_of_int n_params *. 
      Float.log (float_of_int n_obs) in
    
    aic, bic, ll
end

(* State Space *)
module StateSpace = struct
  (* State space model representation *)
  type state_space_model = {
    transition_matrix: Tensor.t;    (* Z matrix *)
    observation_matrix: Tensor.t;   (* Φ matrix *)
    state_noise_cov: Tensor.t;     (* Q matrix *)
    obs_noise_cov: Tensor.t;       (* R matrix *)
    initial_state: Tensor.t;       (* Initial μ *)
    initial_state_cov: Tensor.t;   (* Initial P *)
  }

  (* Kalman filter state *)
  type kalman_state = {
    pred_state: Tensor.t;          (* State prediction *)
    pred_cov: Tensor.t;           (* Prediction covariance *)
    filtered_state: Tensor.t;      (* Filtered state *)
    filtered_cov: Tensor.t;       (* Filtered covariance *)
    log_likelihood: float;        (* Log likelihood *)
  }

  (* Create DNS state space model *)
  let create_dns_model params terms =
    let n_terms = Array.length terms in
    let n_factors = 3 in  (* Level, slope, curvature *)
    
    (* Create observation matrix (Φ) *)
    let phi = nelson_siegel_basis terms.(0) params.lambda in
    let phi = Tensor.of_float_array2 [phi] in
    
    (* Initialize state space matrices *)
    let z = Tensor.eye n_factors ~kind:Float in  (* Default: identity transition *)
    let q = Tensor.eye n_factors ~kind:Float |> 
      Tensor.mul_scalar params.state_noise_var in
    let r = Tensor.eye n_terms ~kind:Float |> 
      Tensor.mul_scalar params.obs_noise_var in
    
    (* Initial state distribution *)
    let init_state = Tensor.zeros [n_factors; 1] ~kind:Float in
    let init_cov = Tensor.eye n_factors ~kind:Float |> 
      Tensor.mul_scalar 1.0 in  (* Large initial uncertainty *)
    
    {
      transition_matrix = z;
      observation_matrix = phi;
      state_noise_cov = q;
      obs_noise_cov = r;
      initial_state = init_state;
      initial_state_cov = init_cov;
    }

  (* Kalman filter prediction step *)
  let predict_step model state =
    (* State prediction *)
    let pred_state = Tensor.mm model.transition_matrix state.filtered_state in
    
    (* Covariance prediction *)
    let pred_cov = Tensor.mm 
      (Tensor.mm model.transition_matrix state.filtered_cov) 
      (Tensor.transpose2 model.transition_matrix) in
    Tensor.add_ pred_cov model.state_noise_cov;
    
    { state with pred_state; pred_cov }

  (* Kalman filter update step *)
  let update_step model state observation =
    (* Innovation computation *)
    let predicted_obs = Tensor.mm model.observation_matrix state.pred_state in
    let innovation = Tensor.sub observation predicted_obs in
    
    (* Innovation covariance *)
    let s = Tensor.mm 
      (Tensor.mm model.observation_matrix state.pred_cov)
      (Tensor.transpose2 model.observation_matrix) in
    Tensor.add_ s model.obs_noise_cov;
    
    (* Validate innovation covariance *)
    validate_positive_definite s "innovation covariance";
    
    (* Kalman gain *)
    let k = Tensor.mm 
      (Tensor.mm state.pred_cov (Tensor.transpose2 model.observation_matrix))
      (Tensor.inverse s) in
    
    (* State update *)
    let filtered_state = Tensor.add state.pred_state 
      (Tensor.mm k innovation) in
    
    (* Covariance update *)
    let i = Tensor.eye (Tensor.shape state.pred_cov |> List.hd) ~kind:Float in
    let ki = Tensor.mm k model.observation_matrix in
    let filtered_cov = Tensor.mm 
      (Tensor.sub i ki) state.pred_cov in
    
    (* Log likelihood computation *)
    let n = Tensor.shape observation |> List.hd in
    let log_det_s = Tensor.logdet s in
    let weighted_innovation = Tensor.mm 
      (Tensor.mm (Tensor.transpose2 innovation) (Tensor.inverse s))
      innovation in
    let ll = -0.5 *. (float_of_int n *. Float.log (2. *. Float.pi) +.
                      log_det_s +.
                      Tensor.to_float0 weighted_innovation) in
    
    { state with
      filtered_state;
      filtered_cov;
      log_likelihood = state.log_likelihood +. ll
    }

  (* Full Kalman filter *)
  let kalman_filter model observations =
    let n_obs = Array.length observations in
    let initial_state = {
      pred_state = model.initial_state;
      pred_cov = model.initial_state_cov;
      filtered_state = model.initial_state;
      filtered_cov = model.initial_state_cov;
      log_likelihood = 0.0;
    } in
    
    (* Process each observation *)
    let rec process_obs state idx =
      if idx >= n_obs then state
      else
        let predicted = predict_step model state in
        let updated = update_step model predicted 
          (Tensor.of_float_array1 observations.(idx) 
             [Array.length observations.(idx)]) in
        process_obs updated (idx + 1)
    in
    
    process_obs initial_state 0
end

(* Dynamic Gaussian Process *)
module DynamicGP = struct
  type dynamic_gp_state = {
    model: gp_model;
    posterior_mean: Tensor.t option;
    posterior_cov: Tensor.t option;
  }

  (* Create initial state *)
  let create_initial_state model =
    {
      model;
      posterior_mean = None;
      posterior_cov = None;
    }

  (* Update state with new observation *)
  let update_state state observation =
    match observation with
    | None -> state
    | Some y ->
        validate_matrix_properties y "observation";
        
        (* Get dimensions *)
        let n = Tensor.shape y |> List.hd in
        
        (* Compute prior covariance *)
        let prior_cov = match state.posterior_cov with
          | None -> Tensor.empty [n; n] ~kind:Float
          | Some cov -> cov in
        
        (* Add system noise *)
        let noise_diag = Tensor.eye n ~kind:Float in
        Tensor.mul_scalar_ noise_diag state.model.params.noise_variance;
        let prior_cov = Tensor.add prior_cov noise_diag in
        
        (* Compute Kalman gain *)
        let s = Tensor.add prior_cov noise_diag in
        let l = Tensor.cholesky s in
        let k = Tensor.triangular_solve prior_cov ~upper:false ~a:l |> fst in
        let k = Tensor.triangular_solve k ~upper:true 
          ~a:(Tensor.transpose2 l) |> fst in
        
        (* Update mean *)
        let innovation = match state.posterior_mean with
          | None -> y
          | Some mean -> Tensor.sub y mean in
        
        let new_mean = match state.posterior_mean with
          | None -> Tensor.mm k innovation
          | Some mean -> Tensor.add mean (Tensor.mm k innovation) in
        
        (* Update covariance *)
        let i = Tensor.eye n ~kind:Float in
        let new_cov = Tensor.mm (Tensor.sub i k) prior_cov in
        
        { state with
          posterior_mean = Some new_mean;
          posterior_cov = Some new_cov;
        }

  (* Predict using dynamic GP *)
  let predict state x_test t =
    match (state.posterior_mean, state.posterior_cov) with
    | None, _ -> 
        (* Initial prediction without prior data *)
        let n_test = Tensor.shape x_test |> List.hd in
        zeros [n_test], zeros [n_test; n_test]
    | Some mean, Some cov ->
        validate_matrix_properties mean "posterior mean";
        validate_matrix_properties cov "posterior covariance";
        
        (* Compute cross-covariances *)
        let k_test_train = compute_kernel_matrix x_test 
          (Tensor.ones [Tensor.shape mean |> List.hd] ~kind:Float)
          state.model.params state.model.kernel_type t in
        
        (* Compute predictive distribution *)
        let pred_mean = Tensor.mm k_test_train 
          (Tensor.mm (Tensor.inverse cov) mean) in
        
        let k_test_test = compute_kernel_matrix x_test x_test
          state.model.params state.model.kernel_type t in
        let pred_cov = Tensor.sub k_test_test
          (Tensor.mm k_test_train 
             (Tensor.mm (Tensor.inverse cov) 
                (Tensor.transpose2 k_test_train))) in
        
        pred_mean, pred_cov

    Tensor.set mat [0; 0] 1.0;
    Tensor.set mat [0; 1] (-2.0);
    Tensor.set mat [0; 2] 1.0;
    
    Tensor.set mat [n-1; n-3] 1.0;
    Tensor.set mat [n-1; n-2] (-2.0);
    Tensor.set mat [n-1; n-1] 1.0;
    
    Tensor.div_scalar mat (h *. h)

  (* Integration methods *)
  let integrate f a b n method_ =
    let h = (b -. a) /. float_of_int n in
    let points = Array.init (n+1) (fun i -> a +. float_of_int i *. h) in
    match method_ with
    | Trapezoidal ->
        let sum = ref ((f points.(0) +. f points.(n)) /. 2.) in
        for i = 1 to n-1 do
          sum := !sum +. f points.(i)
        done;
        h *. !sum
    | Simpson when n mod 2 = 0 ->
        let sum = ref (f points.(0) +. f points.(n)) in
        for i = 1 to n-1 do
          sum := !sum +. (if i mod 2 = 0 then 2. else 4.) *. f points.(i)
        done;
        h *. !sum /. 3.
    | RectangularLeft ->
        let sum = ref 0. in
        for i = 0 to n-1 do
          sum := !sum +. f points.(i)
        done;
        h *. !sum
    | _ -> failwith "Invalid integration method or parameters"

  (* Penalty matrix computation *)
  let compute_penalty_matrix points h =
    let d2 = d2_operator_matrix points h in
    let d2_t = Tensor.transpose2 d2 in
    Tensor.mm d2_t d2

  (* Basis function implementations *)
  let fourier_basis max_freq period t =
    let omega = 2. *. Float.pi /. period in
    Array.init (2 * max_freq + 1) (fun k ->
      if k = 0 then 1.
      else if k mod 2 = 1 then 
        Float.sin (omega *. float_of_int ((k+1)/2) *. t)
      else 
        Float.cos (omega *. float_of_int (k/2) *. t))

  let exponential_basis rates t =
    Array.map (fun rate -> Float.exp (rate *. t)) rates

  let gaussian_basis centers width t =
    Array.map (fun center ->
      Float.exp (-.(t -. center) *. (t -. center) /. 
        (2. *. width *. width))) centers

  (* Function approximation using basis functions *)
  let approximate_function basis coeffs t =
    let basis_values = basis t in
    Array.fold_left2 (fun acc b c -> acc +. b *. c) 0. 
      basis_values coeffs
end

(* Parameter Learning *)
module ParameterLearning = struct
  (* OLS estimation *)
  let estimate_ols phi y =
    validate_matrix_properties phi "design matrix";
    validate_matrix_properties y "target vector";
    
    let phi_t = Tensor.transpose2 phi in
    let phi_phi = Tensor.mm phi_t phi in
    
    validate_positive_definite phi_phi "OLS normal equations";
    
    let phi_y = Tensor.mm phi_t y in
    Tensor.mm (Tensor.inverse phi_phi) phi_y

  (* Penalized least squares estimation *)
  let estimate_penalized_ls phi y lambda r2 =
    validate_matrix_properties phi "design matrix";
    validate_matrix_properties y "target vector";
    validate_matrix_properties r2 "penalty matrix";
    
    let phi_t = Tensor.transpose2 phi in
    let phi_phi = Tensor.mm phi_t phi in
    let phi_y = Tensor.mm phi_t y in
    let penalty = Tensor.mul_scalar r2 lambda in
    let reg_matrix = Tensor.add phi_phi penalty in
    
    validate_positive_definite reg_matrix "penalized normal equations";
    
    Tensor.mm (Tensor.inverse reg_matrix) phi_y

  (* Bayesian parameter estimation *)
  let estimate_bayesian phi y prior_mean prior_cov noise_var =
    validate_matrix_properties phi "design matrix";
    validate_matrix_properties y "target vector";
    validate_matrix_properties prior_mean "prior mean";
    validate_matrix_properties prior_cov "prior covariance";
    
    let phi_t = Tensor.transpose2 phi in
    let s_inv = Tensor.div_scalar phi_t noise_var in
    let s_inv = Tensor.mm s_inv phi in
    let prior_cov_inv = Tensor.inverse prior_cov in
    
    (* Posterior covariance *)
    let post_cov = Tensor.add s_inv prior_cov_inv in
    validate_positive_definite post_cov "posterior covariance";
    let post_cov = Tensor.inverse post_cov in
    
    (* Posterior mean *)
    let term1 = Tensor.mm s_inv y in
    let term2 = Tensor.mm prior_cov_inv prior_mean in
    let post_mean = Tensor.add term1 term2 in
    let post_mean = Tensor.mm post_cov post_mean in
    
    post_mean, post_cov

  (* Cross-validation for parameter selection *)
  let cross_validate_parameters phi y folds estimate_fn =
    let n = Tensor.shape phi |> List.hd in
    let fold_size = n / folds in
    
    let errors = Array.init folds (fun fold ->
      let start_idx = fold * fold_size in
      let end_idx = min (start_idx + fold_size) n in
      
      (* Split data *)
      let train_phi = Tensor.cat
        [Tensor.narrow phi 0 0 start_idx;
         Tensor.narrow phi 0 end_idx (n - end_idx)] 0 in
      let train_y = Tensor.cat
        [Tensor.narrow y 0 0 start_idx;
         Tensor.narrow y 0 end_idx (n - end_idx)] 0 in
      
      let test_phi = Tensor.narrow phi 0 start_idx (end_idx - start_idx) in
      let test_y = Tensor.narrow y 0 start_idx (end_idx - start_idx) in
      
      (* Estimate parameters *)
      let params = estimate_fn train_phi train_y in
      
      (* Compute validation error *)
      let pred = Tensor.mm test_phi params in
      let error = Tensor.sub pred test_y in
      Tensor.mean (Tensor.pow error (Tensor.float_vec [2.0])) |>
      Tensor.to_float0
    ) in
    
    Array.fold_left (+.) 0. errors /. float_of_int folds
end