open Torch

(* Convert vector to diagonal matrix *)
let diag v = 
  let n = Tensor.shape v |> List.hd in
  let m = Tensor.zeros [n; n] in
  for i = 0 to n - 1 do
    Tensor.set m [i; i] (Tensor.get v [i])
  done;
  m

(* Log-sum-exp trick for numerical stability *)
let log_sum_exp x =
  let max_x = Tensor.max x [0] false in
  let shifted = Tensor.(-) x max_x in
  let sum_exp = Tensor.sum (Tensor.exp shifted) [0] false in
  Tensor.(+) max_x (Tensor.log sum_exp)

(* Create a block diagonal matrix from a list of matrices *)
let block_diag mats =
  let total_rows = List.fold_left (fun acc m -> acc + (Tensor.shape m |> List.hd)) 0 mats in
  let total_cols = List.fold_left (fun acc m -> acc + (Tensor.shape m |> List.tl |> List.hd)) 0 mats in
  let result = Tensor.zeros [total_rows; total_cols] in
  
  let row_offset = ref 0 in
  let col_offset = ref 0 in
  
  List.iter (fun m ->
    let rows = Tensor.shape m |> List.hd in
    let cols = Tensor.shape m |> List.tl |> List.hd in
    
    for i = 0 to rows - 1 do
      for j = 0 to cols - 1 do
        let v = Tensor.get m [i; j] in
        Tensor.set result [!row_offset + i; !col_offset + j] v
      done
    done;
    
    row_offset := !row_offset + rows;
    col_offset := !col_offset + cols
  ) mats;
  
  result

(* Extract half-vectorization (vech) of a symmetric matrix *)
let vech m =
  let n = Tensor.shape m |> List.hd in
  let size = n * (n + 1) / 2 in
  let result = Tensor.zeros [size] in
  
  let idx = ref 0 in
  for i = 0 to n - 1 do
    for j = 0 to i do
      Tensor.set result [!idx] (Tensor.get m [i; j]);
      incr idx
    done
  done;
  
  result

(* Inverse of vech: convert half-vectorization to symmetric matrix *)
let vech_to_matrix v =
  let size = Tensor.shape v |> List.hd in
  let n = int_of_float ((-1.0 +. sqrt (1.0 +. 8.0 *. float_of_int size)) /. 2.0) in
  let m = Tensor.zeros [n; n] in
  
  let idx = ref 0 in
  for i = 0 to n - 1 do
    for j = 0 to i do
      let val_ = Tensor.get v [!idx] in
      Tensor.set m [i; j] val_;
      Tensor.set m [j; i] val_;  (* Ensure symmetry *)
      incr idx
    done
  done;
  
  m

(* Vectorize a matrix (column-wise) *)
let vec m =
  let shape = Tensor.shape m in
  let rows = List.hd shape in
  let cols = List.nth shape 1 in
  Tensor.reshape m [rows * cols]

(* Reshape a vector to a matrix *)
let vec_to_matrix v rows cols =
  Tensor.reshape v [rows; cols]

(* Compute Cholesky decomposition for positive definite matrix *)
let cholesky m =
  Tensor.cholesky m Upper

(* Compute log determinant of a matrix from its Cholesky factor *)
let log_det_from_cholesky l =
  let diag_elements = Tensor.diagonal l ~dim1:0 ~dim2:1 ~offset:0 in
  let log_diag = Tensor.log (Tensor.abs diag_elements) in
  Tensor.sum log_diag [0] false |> Tensor.mul_scalar (Tensor.of_float 2.0)

(* Convert precision matrix to covariance matrix *)
let precision_to_covariance prec =
  (* Use Cholesky and triangular solve for numerical stability *)
  let l = cholesky prec in
  let eye = Tensor.eye (Tensor.shape prec |> List.hd) in
  let l_inv = Tensor.triangular_solve l eye ~upper:true ~transpose:false ~unitriangular:false in
  Tensor.matmul (Tensor.transpose l_inv ~dim0:0 ~dim1:1) l_inv

(* Calculate KL divergence between two multivariate normal distributions *)
let kl_mvn_from_params mu1 l1 mu2 l2 =
  let n = Tensor.shape mu1 |> List.hd in
  let n_float = float_of_int n in
  
  (* Calculate inverse of covariance matrix 2 using Cholesky factor *)
  let l2_inv = Tensor.triangular_solve l2 (Tensor.eye n) ~upper:true ~transpose:false ~unitriangular:false in
  let prec2 = Tensor.matmul (Tensor.transpose l2_inv ~dim0:0 ~dim1:1) l2_inv in
  
  (* Mean difference *)
  let mu_diff = Tensor.(-) mu1 mu2 in
  
  (* Trace term *)
  let cov1 = Tensor.matmul (Tensor.transpose l1 ~dim0:0 ~dim1:1) l1 in
  let trace_term = Tensor.sum (Tensor.mul (Tensor.matmul cov1 prec2) (Tensor.ones [1; 1])) [0; 1] false in
  
  (* Mahalanobis distance *)
  let mahalanobis = Tensor.matmul (Tensor.matmul (Tensor.reshape mu_diff [1; n]) prec2) 
                     (Tensor.reshape mu_diff [n; 1]) in
  
  (* Log determinant terms *)
  let logdet1 = log_det_from_cholesky l1 in
  let logdet2 = log_det_from_cholesky l2 in
  
  (* KL formula: 0.5 * (trace + mahalanobis - n + logdet2 - logdet1) *)
  let result = Tensor.(+) trace_term mahalanobis in
  let result = Tensor.(-) result (Tensor.of_float n_float) in
  let result = Tensor.(+) result (Tensor.(-) logdet2 logdet1) in
  Tensor.mul_scalar result (Tensor.of_float 0.5)

(* Create a mask for sparse Cholesky factor *)
let create_sparse_cholesky_mask n_latent n_global =
  let total_size = n_latent + n_global in
  let mask = Tensor.zeros [total_size; total_size] in
  
  (* Set up the block structure *)
  for i = 0 to total_size - 1 do
    for j = 0 to i do
      if i < n_latent then
        (* Diagonal block for latent variables *)
        if j == i then
          Tensor.set mask [i; j] (Tensor.of_float 1.0)
      else
        (* Last n_global rows *)
        Tensor.set mask [i; j] (Tensor.of_float 1.0)
    done
  done;
  
  mask

(* Create a mask for state space model Cholesky factor *)
let create_state_space_cholesky_mask n_states n_global =
  let total_size = n_states + n_global in
  let mask = Tensor.zeros [total_size; total_size] in
  
  (* Set up the block tridiagonal structure *)
  for i = 0 to total_size - 1 do
    for j = 0 to i do
      if i < n_states then
        (* Diagonal elements *)
        if j == i then
          Tensor.set mask [i; j] (Tensor.of_float 1.0)
        (* Off-diagonal elements for state transitions *)
        else if i > 0 && j == (i-1) then
          Tensor.set mask [i; j] (Tensor.of_float 1.0)
      else
        (* Last n_global rows - fully connected to all states *)
        Tensor.set mask [i; j] (Tensor.of_float 1.0)
    done
  done;
  
  mask

(* Apply sparsity mask to a matrix *)
let apply_mask m mask =
  Tensor.mul m mask

(* ADAM optimizer parameters *)
type adam_params = {
  beta1: float;
  beta2: float;
  epsilon: float;
  learning_rate: float;
}

(* Default ADAM parameters *)
let default_adam_params = {
  beta1 = 0.9;
  beta2 = 0.999;
  epsilon = 1e-8;
  learning_rate = 0.001;
}

(* ADAM optimizer state *)
type adam_state = {
  m: Tensor.t; (* First moment *)
  v: Tensor.t; (* Second moment *)
  t: int;      (* Timestep *)
}

(* Initialize ADAM state *)
let init_adam_state shape =
  {
    m = Tensor.zeros shape;
    v = Tensor.zeros shape;
    t = 0;
  }

(* ADAM update step *)
let adam_update params state grad =
  let t = state.t + 1 in
  let beta1_t = Tensor.of_float (params.beta1 ** float_of_int t) in
  let beta2_t = Tensor.of_float (params.beta2 ** float_of_int t) in
  
  (* Update biased first moment estimate *)
  let m = Tensor.(+) 
    (Tensor.mul_scalar state.m (Tensor.of_float params.beta1))
    (Tensor.mul_scalar grad (Tensor.of_float (1.0 -. params.beta1))) in
  
  (* Update biased second raw moment estimate *)
  let v = Tensor.(+)
    (Tensor.mul_scalar state.v (Tensor.of_float params.beta2))
    (Tensor.mul_scalar (Tensor.mul grad grad) (Tensor.of_float (1.0 -. params.beta2))) in
  
  (* Bias correction *)
  let m_hat = Tensor.(/) m (Tensor.sub (Tensor.of_float 1.0) beta1_t) in
  let v_hat = Tensor.(/) v (Tensor.sub (Tensor.of_float 1.0) beta2_t) in
  
  (* Update parameters *)
  let step_size = Tensor.of_float params.learning_rate in
  let denom = Tensor.(+) (Tensor.sqrt v_hat) (Tensor.of_float params.epsilon) in
  let update = Tensor.(/) (Tensor.mul step_size m_hat) denom in
  
  (* Return new state and update *)
  ({ m; v; t }, update)

(* Convergence assessment *)
let assess_convergence prev_value curr_value tol =
  let abs_diff = abs_float (curr_value -. prev_value) in
  let rel_diff = abs_diff /. (abs_float prev_value +. 1e-10) in
  abs_diff < tol || rel_diff < tol

(* Calculate RMSE between tensors *)
let rmse a b =
  let diff = Tensor.(-) a b in
  let squared = Tensor.mul diff diff in
  let mean_squared = Tensor.mean squared ~dim:[0] ~keepdim:false in
  Tensor.sqrt mean_squared |> Tensor.to_float0_exn

(* Generate reparameterized sample from Gaussian component *)
let reparameterize_gaussian mu l_chol =
  let n = Tensor.shape mu |> List.hd in
  let epsilon = Tensor.randn [n] in
  let epsilon_reshaped = Tensor.reshape epsilon [n; 1] in
  
  (* Solve L'x = ε *)
  let l_inv_eps = Tensor.triangular_solve 
                   l_chol
                   epsilon_reshaped
                   ~upper:true
                   ~transpose:false
                   ~unitriangular:false in
  
  Tensor.(+) mu (Tensor.reshape l_inv_eps [n])

(* Reparameterization-based gradient for mean *)
let reparam_grad_mean mu l_chol log_prob_fn n_samples =
  let n = Tensor.shape mu |> List.hd in
  let gradients = Tensor.zeros [n] in
  
  (* Make a copy of mu that requires gradients *)
  let mu_grad = Tensor.copy mu ~requires_grad:true in
  
  for _ = 1 to n_samples do
    (* Sample noise *)
    let epsilon = Tensor.randn [n] in
    let epsilon_reshaped = Tensor.reshape epsilon [n; 1] in
    
    (* Generate sample using reparameterization *)
    let l_inv_eps = Tensor.triangular_solve 
                    l_chol
                    epsilon_reshaped
                    ~upper:true
                    ~transpose:false
                    ~unitriangular:false in
    
    let sample = Tensor.(+) mu_grad (Tensor.reshape l_inv_eps [n]) in
    
    (* Compute log probability and its gradient *)
    let log_prob = log_prob_fn sample in
    let grad = Tensor.grad log_prob ~inputs:[mu_grad] in
    
    (* Accumulate gradients *)
    Tensor.(+=) gradients (List.hd grad |> Tensor.copy)
  done;
  
  (* Average gradients *)
  Tensor.(/) gradients (Tensor.of_float (float_of_int n_samples))

(* Reparameterization-based gradient for Cholesky factor *)
let reparam_grad_cholesky mu l_chol log_prob_fn n_samples =
  let n = Tensor.shape mu |> List.hd in
  let gradients = Tensor.zeros_like l_chol in
  
  (* Make a copy of l_chol that requires gradients *)
  let l_chol_grad = Tensor.copy l_chol ~requires_grad:true in
  
  for _ = 1 to n_samples do
    (* Sample noise *)
    let epsilon = Tensor.randn [n] in
    let epsilon_reshaped = Tensor.reshape epsilon [n; 1] in
    
    (* Generate sample using reparameterization *)
    let l_inv_eps = Tensor.triangular_solve 
                    l_chol_grad
                    epsilon_reshaped
                    ~upper:true
                    ~transpose:false
                    ~unitriangular:false in
    
    let sample = Tensor.(+) mu (Tensor.reshape l_inv_eps [n]) in
    
    (* Compute log probability and its gradient *)
    let log_prob = log_prob_fn sample in
    let grad = Tensor.grad log_prob ~inputs:[l_chol_grad] in
    
    (* Now implement the specific form *)
    let grad_l = List.hd grad in
    
    (* Calculate grad_L' * epsilon * (grad_log p - grad_log q)' * L' *)
    let grad_term = Tensor.matmul 
                     (Tensor.neg (Tensor.transpose l_inv_eps ~dim0:0 ~dim1:1))
                     (Tensor.matmul 
                       (Tensor.reshape (Tensor.grad log_prob ~inputs:[sample] |> List.hd) [1; n])
                       (Tensor.transpose l_chol_grad ~dim0:0 ~dim1:1)) in
    
    (* Accumulate gradients *)
    Tensor.(+=) gradients (Tensor.(+) grad_l grad_term |> Tensor.copy)
  done;
  
  (* Average gradients *)
  Tensor.(/) gradients (Tensor.of_float (float_of_int n_samples))

(* Control variates for variance reduction in gradient estimation *)
let control_variate_gradient gradient_estimates observation_function =
  let n_samples = List.length gradient_estimates in
  
  (* Compute average gradient *)
  let avg_gradient = List.fold_left (fun acc g -> Tensor.(+) acc g) 
                       (Tensor.zeros_like (List.hd gradient_estimates)) 
                       gradient_estimates in
  let avg_gradient = Tensor.(/) avg_gradient (Tensor.of_float (float_of_int n_samples)) in
  
  (* Compute average observation *)
  let observations = List.map observation_function gradient_estimates in
  let avg_observation = List.fold_left (+.) 0.0 observations /. float_of_int n_samples in
  
  (* Compute covariance between gradients and observations *)
  let cov_go = ref (Tensor.zeros_like avg_gradient) in
  let var_o = ref 0.0 in
  
  List.iter2 (fun g o ->
    let g_centered = Tensor.(-) g avg_gradient in
    let o_centered = o -. avg_observation in
    Tensor.(+=) !cov_go (Tensor.mul_scalar g_centered (Tensor.of_float o_centered));
    var_o := !var_o +. (o_centered *. o_centered)
  ) gradient_estimates observations;
  
  let cov_go = Tensor.(/) !cov_go (Tensor.of_float (float_of_int n_samples)) in
  let var_o = !var_o /. float_of_int n_samples in
  
  (* Compute optimal control variate coefficient *)
  let c = if var_o < 1e-10 then 
    Tensor.zeros_like cov_go 
  else 
    Tensor.(/) cov_go (Tensor.of_float var_o) in
  
  (* Apply control variate to reduce variance *)
  let gradient_cv = List.map2 (fun g o ->
    Tensor.(-) g (Tensor.mul_scalar c (Tensor.of_float (o -. avg_observation)))
  ) gradient_estimates observations in
  
  (* Compute average gradient with control variate *)
  let avg_gradient_cv = List.fold_left (fun acc g -> Tensor.(+) acc g) 
                         (Tensor.zeros_like (List.hd gradient_cv)) 
                         gradient_cv in
  
  Tensor.(/) avg_gradient_cv (Tensor.of_float (float_of_int n_samples))

(* Variance reduction technique from Ranganath et al. (2014) *)
let ranganath_control_variate mu l_chol log_joint_fn log_q_fn n_samples =
  let n = Tensor.shape mu |> List.hd in
  let gradients = ref [] in
  let observations = ref [] in
  
  (* Generate samples *)
  for _ = 1 to n_samples do
    (* Sample from the distribution *)
    let sample = reparameterize_gaussian mu l_chol in
    
    (* Compute log probabilities *)
    let log_joint = log_joint_fn sample in
    let log_q = log_q_fn sample in
    
    (* ELBO component *)
    let elbo_term = Tensor.(-) log_joint log_q in
    
    (* Score function: ∇_θ log q(z|θ) *)
    let score = Tensor.grad log_q ~inputs:[mu] |> List.hd in
    
    (* Gradient estimate: (log p(x,z) - log q(z|θ)) * ∇_θ log q(z|θ) *)
    let gradient = Tensor.mul_scalar score elbo_term in
    
    gradients := gradient :: !gradients;
    observations := (Tensor.to_float0_exn elbo_term) :: !observations
  done;
  
  control_variate_gradient (List.rev !gradients) (fun _ -> List.hd !observations)

(* Combine multiple variance reduction techniques *)
let combined_variance_reduction mu l_chol log_joint_fn log_q_fn n_samples =
  (* First use reparameterization trick for mean parameters *)
  let mean_grad = reparam_grad_mean mu l_chol 
                 (fun sample -> Tensor.(-) (log_joint_fn sample) (log_q_fn sample))
                 n_samples in
  
  (* Use combination of reparameterization and control variates for Cholesky factor *)
  let chol_grad_samples = ref [] in
  let log_probs = ref [] in
  
  for _ = 1 to n_samples do
    (* Sample from the distribution *)
    let sample = reparameterize_gaussian mu l_chol in
    
    (* Compute log probabilities *)
    let log_joint = log_joint_fn sample in
    let log_q = log_q_fn sample in
    let elbo_term = Tensor.(-) log_joint log_q in
    
    (* Reparameterization gradient for Cholesky factor *)
    let epsilon = Tensor.randn [n] in
    let epsilon_reshaped = Tensor.reshape epsilon [n; 1] in
    
    let l_inv_eps = Tensor.triangular_solve 
                   l_chol
                   epsilon_reshaped
                   ~upper:true
                   ~transpose:false
                   ~unitriangular:false in
    
    let grad_term = Tensor.matmul 
                    (Tensor.neg (Tensor.transpose l_inv_eps ~dim0:0 ~dim1:1))
                    (Tensor.matmul 
                      (Tensor.reshape (Tensor.grad elbo_term ~inputs:[sample] |> List.hd) [1; n])
                      (Tensor.transpose l_chol ~dim0:0 ~dim1:1)) in
    
    chol_grad_samples := grad_term :: !chol_grad_samples;
    log_probs := (Tensor.to_float0_exn elbo_term) :: !log_probs
  done;
  
  (* Apply control variates to Cholesky factor gradients *)
  let chol_grad = control_variate_gradient 
                  (List.rev !chol_grad_samples)
                  (fun _ -> List.hd !log_probs) in
  
  (mean_grad, chol_grad)

(* Initialize mean vector for a new component *)
let initialize_mean ~base_mean ~n_latent ~n_global ~strategy ~scale =
  match strategy with
  | `Random -> 
      (* Random perturbation around base mean *)
      Tensor.(+) base_mean (Tensor.mul_scalar (Tensor.randn_like base_mean) (Tensor.of_float scale))
  
  | `KMeans ~data ~k ->
      (* Use K-means to initialize from data samples *)
      let n_data = Tensor.shape data |> List.hd in
      let total_dim = n_latent + n_global in
      let centers = Tensor.zeros [k; total_dim] in
      
      (* Initialize centers randomly from data *)
      for i = 0 to k - 1 do
        let idx = Random.int n_data in
        let data_point = Tensor.select data ~dim:0 ~index:idx in
        for j = 0 to total_dim - 1 do
          Tensor.set centers [i; j] (Tensor.get data_point [j])
        done
      done;
      
      (* Run K-means for a few iterations *)
      let max_iter = 5 in
      for _ = 1 to max_iter do
        (* Assign points to nearest center *)
        let assignments = Array.make n_data 0 in
        for i = 0 to n_data - 1 do
          let data_point = Tensor.select data ~dim:0 ~index:i in
          let mut_min_dist = ref Float.max_float in
          let mut_min_idx = ref 0 in
          
          for j = 0 to k - 1 do
            let center = Tensor.select centers ~dim:0 ~index:j in
            let diff = Tensor.(-) data_point center in
            let dist = Tensor.sum (Tensor.mul diff diff) ~dim:[0] ~keepdim:false |> Tensor.to_float0_exn in
            
            if dist < !mut_min_dist then begin
              mut_min_dist := dist;
              mut_min_idx := j
            end
          done;
          
          assignments.(i) <- !mut_min_idx
        done;
        
        (* Update centers *)
        let new_centers = Tensor.zeros [k; total_dim] in
        let counts = Array.make k 0 in
        
        for i = 0 to n_data - 1 do
          let cluster = assignments.(i) in
          let data_point = Tensor.select data ~dim:0 ~index:i in
          
          for j = 0 to total_dim - 1 do
            let current = Tensor.get new_centers [cluster; j] |> Tensor.to_float0_exn in
            let value = Tensor.get data_point [j] |> Tensor.to_float0_exn in
            Tensor.set new_centers [cluster; j] (Tensor.of_float (current +. value))
          done;
          
          counts.(cluster) <- counts.(cluster) + 1
        done;
        
        for i = 0 to k - 1 do
          if counts.(i) > 0 then
            for j = 0 to total_dim - 1 do
              let current = Tensor.get new_centers [i; j] |> Tensor.to_float0_exn in
              Tensor.set new_centers [i; j] (Tensor.of_float (current /. float_of_int counts.(i)))
            done
        done;
        
        centers <- new_centers
      done;
      
      (* Return the center with the most assigned points *)
      let best_cluster = ref 0 in
      let max_count = ref 0 in
      Array.iteri (fun i count ->
        if count > !max_count then begin
          max_count := count;
          best_cluster := i
        end
      ) (Array.make k 0);
      
      Tensor.select centers ~dim:0 ~index:!best_cluster
  
  | `Factor ->
      (* Use factor analysis to initialize *)
      let n_components = min 10 (n_latent + n_global) in
      
      (* Create random factor loadings *)
      let loadings = Tensor.randn [n_latent + n_global; n_components] in
      
      (* Create random factor scores *)
      let factors = Tensor.randn [n_components] in
      
      (* Compute mean as base_mean + loadings * factors *)
      let perturbation = Tensor.matmul loadings (Tensor.reshape factors [n_components; 1]) in
      
      Tensor.(+) base_mean (Tensor.reshape perturbation [n_latent + n_global] 
                          |> Tensor.mul_scalar (Tensor.of_float scale))

(* Initialize Cholesky factor for a new component *)
let initialize_cholesky ~base_chol ~n_latent ~n_global ~strategy ~scale =
  match strategy with
  | `Identity ->
      (* Start with the identity matrix but keep the sparsity pattern *)
      let total_dim = n_latent + n_global in
      let mask = create_sparse_cholesky_mask n_latent n_global in
      
      (* Create identity-like matrix with some noise *)
      let eye = Tensor.eye total_dim in
      let noise = Tensor.mul_scalar (Tensor.randn [total_dim; total_dim]) (Tensor.of_float scale) in
      let result = Tensor.(+) eye noise in
      
      (* Apply mask to maintain sparsity pattern *)
      apply_mask result mask
  
  | `Perturb ->
      (* Perturb the base Cholesky factor *)
      let mask = create_sparse_cholesky_mask n_latent n_global in
      let noise = Tensor.mul_scalar (Tensor.randn_like base_chol) (Tensor.of_float scale) in
      let result = Tensor.(+) base_chol noise in
      
      (* Apply mask to maintain sparsity pattern *)
      apply_mask result mask
  
  | `Diagonal diag_value ->
      (* Start with a diagonal matrix with given values *)
      let total_dim = n_latent + n_global in
      let mask = create_sparse_cholesky_mask n_latent n_global in
      
      (* Create diagonal matrix *)
      let diag_mat = Tensor.eye total_dim |> Tensor.mul_scalar (Tensor.of_float diag_value) in
      let noise = Tensor.mul_scalar (Tensor.randn [total_dim; total_dim]) (Tensor.of_float (scale *. 0.1)) in
      let result = Tensor.(+) diag_mat noise in
      
      (* Apply mask to maintain sparsity pattern *)
      apply_mask result mask
  
  | `StateSpace ->
      (* Special initialization for state space models *)
      let total_dim = n_latent + n_global in
      let mask = create_state_space_cholesky_mask (n_latent / 1) n_global in
      
      (* Create base matrix *)
      let diag_mat = Tensor.eye total_dim in
      
      (* Add AR(1)-like structure for state transitions *)
      let n_time_points = n_latent in
      for t = 1 to n_time_points - 1 do
        (* Add transition from previous state with small value *)
        Tensor.set diag_mat [t; t-1] (Tensor.of_float 0.9)
      done;
      
      let noise = Tensor.mul_scalar (Tensor.randn [total_dim; total_dim]) (Tensor.of_float scale) in
      let result = Tensor.(+) diag_mat noise in
      
      (* Apply mask to maintain sparsity pattern *)
      apply_mask result mask

(* Gaussian component parameters for variational approximation *)
type gaussian_component = {
  mu: Tensor.t;           (* Mean vector *)
  l_chol: Tensor.t;       (* Cholesky factor of precision matrix *)
  mu_g: Tensor.t;         (* Global parameters mean *)
  mu_b: Tensor.t;         (* Local parameters (latent variables) mean *)
  l_g: Tensor.t;          (* Cholesky factor for global parameters *)
  l_b: Tensor.t list;     (* Cholesky factors for each latent variable block *)
  l_gb: Tensor.t list;    (* Cross-covariance terms *)
  l_tilde: Tensor.t list option; (* State transition matrices for state space models (optional) *)
}

(* Gaussian mixture distribution *)
type gaussian_mixture = {
  components: gaussian_component array;
  weights: Tensor.t;       (* Mixture weights *)
  n_components: int;
  n_global: int;           (* Number of global parameters *)
  n_latent: int;           (* Number of latent variables *)
  n_subjects: int;         (* Number of subjects/data points *)
  latent_dim: int;         (* Dimension of each latent variable *)
}

(* Initialize a Gaussian component with the proper structure *)
let init_gaussian_component ~n_global ~n_subjects ~latent_dim ~use_sparse_cholesky =
  let n_latent = n_subjects * latent_dim in
  let total_dim = n_global + n_latent in
  
  (* Initialize means with small random values *)
  let mu = Tensor.randn [total_dim] ~dtype:(Torch_core.Kind.T Float) in
  let mu_g = Tensor.narrow mu ~dim:0 ~start:(n_latent) ~length:n_global in
  let mu_b = Tensor.narrow mu ~dim:0 ~start:0 ~length:n_latent in
  
  (* Initialize Cholesky factors *)
  let l_chol = if use_sparse_cholesky then
    let mask = create_sparse_cholesky_mask n_latent n_global in
    (* Initialize with identity-like structure but apply sparsity mask *)
    let eye = Tensor.eye total_dim in
    let random_noise = Tensor.mul_scalar (Tensor.randn [total_dim; total_dim]) (Tensor.of_float 0.01) in
    let initial_l = Tensor.(+) eye random_noise in
    apply_mask initial_l mask
  else
    (* Full Cholesky without sparsity *)
    let eye = Tensor.eye total_dim in
    Tensor.(+) eye (Tensor.mul_scalar (Tensor.randn [total_dim; total_dim]) (Tensor.of_float 0.01))
  in
  
  (* Extract sub-matrices *)
  let l_g = Tensor.narrow l_chol ~dim:0 ~start:(n_latent) ~length:n_global |>
           fun x -> Tensor.narrow x ~dim:1 ~start:(n_latent) ~length:n_global in
  
  (* Extract Cholesky factors for each latent variable block *)
  let l_b = List.init n_subjects (fun i ->
    let start = i * latent_dim in
    Tensor.narrow l_chol ~dim:0 ~start ~length:latent_dim |>
    fun x -> Tensor.narrow x ~dim:1 ~start ~length:latent_dim
  ) in
  
  (* Extract cross-covariance terms *)
  let l_gb = List.init n_subjects (fun i ->
    let start = i * latent_dim in
    Tensor.narrow l_chol ~dim:0 ~start:(n_latent) ~length:n_global |>
    fun x -> Tensor.narrow x ~dim:1 ~start ~length:latent_dim
  ) in
  
  { mu; l_chol; mu_g; mu_b; l_g; l_b; l_gb; l_tilde = None }

(* Initialize a state space Gaussian component *)
let init_state_space_component ~n_time_points ~latent_dim ~global_dim ~use_sparse_cholesky =
  let n_latent = n_time_points * latent_dim in
  let total_dim = global_dim + n_latent in
  
  (* Initialize means with small random values *)
  let mu = Tensor.randn [total_dim] ~dtype:(Torch_core.Kind.T Float) in
  let mu_g = Tensor.narrow mu ~dim:0 ~start:(n_latent) ~length:global_dim in
  let mu_b = Tensor.narrow mu ~dim:0 ~start:0 ~length:n_latent in
  
  (* Initialize Cholesky factor with state space structure *)
  let l_chol = if use_sparse_cholesky then
    let mask = create_state_space_cholesky_mask n_latent global_dim in
    (* Initialize with AR(1)-like structure *)
    let diag_mat = Tensor.eye total_dim in
    
    (* Add transition structure *)
    for t = 1 to n_time_points - 1 do
      let src_idx = (t-1) * latent_dim in
      let dst_idx = t * latent_dim in
      for i = 0 to latent_dim - 1 do
        for j = 0 to latent_dim - 1 do
          if i = j then
            Tensor.set diag_mat [dst_idx + i; src_idx + j] (Tensor.of_float 0.9)
        done
      done
    done;
    
    let random_noise = Tensor.mul_scalar (Tensor.randn [total_dim; total_dim]) (Tensor.of_float 0.01) in
    let initial_l = Tensor.(+) diag_mat random_noise in
    apply_mask initial_l mask
  else
    (* Full Cholesky without sparsity *)
    let eye = Tensor.eye total_dim in
    Tensor.(+) eye (Tensor.mul_scalar (Tensor.randn [total_dim; total_dim]) (Tensor.of_float 0.01))
  in
  
  (* Extract sub-matrices *)
  let l_g = Tensor.narrow l_chol ~dim:0 ~start:(n_latent) ~length:global_dim |>
           fun x -> Tensor.narrow x ~dim:1 ~start:(n_latent) ~length:global_dim in
  
  (* Extract Cholesky factors for each state *)
  let l_b = List.init n_time_points (fun i ->
    let start = i * latent_dim in
    Tensor.narrow l_chol ~dim:0 ~start ~length:latent_dim |>
    fun x -> Tensor.narrow x ~dim:1 ~start ~length:latent_dim
  ) in
  
  (* Extract transition matrices *)
  let l_tilde = List.init (n_time_points - 1) (fun i ->
    let curr_start = (i + 1) * latent_dim in
    let prev_start = i * latent_dim in
    Tensor.narrow l_chol ~dim:0 ~start:curr_start ~length:latent_dim |>
    fun x -> Tensor.narrow x ~dim:1 ~start:prev_start ~length:latent_dim
  ) in
  
  (* Extract cross-covariance terms *)
  let l_gb = List.init n_time_points (fun i ->
    let start = i * latent_dim in
    Tensor.narrow l_chol ~dim:0 ~start:(n_latent) ~length:global_dim |>
    fun x -> Tensor.narrow x ~dim:1 ~start ~length:latent_dim
  ) in
  
  { mu; l_chol; mu_g; mu_b; l_g; l_b; l_gb; l_tilde = Some l_tilde }

(* Initialize a Gaussian mixture distribution *)
let init_gaussian_mixture ~n_components ~n_global ~n_subjects ~latent_dim ~use_sparse_cholesky =
  let n_latent = n_subjects * latent_dim in
  
  (* Initialize components *)
  let components = Array.init n_components (fun _ ->
    init_gaussian_component ~n_global ~n_subjects ~latent_dim ~use_sparse_cholesky
  ) in
  
  (* Initialize weights with uniform distribution *)
  let weights = Tensor.ones [n_components] |> 
                Tensor.div_scalar (Tensor.of_float (float_of_int n_components)) in
  
  { components; weights; n_components; n_global; n_latent; n_subjects; latent_dim }

(* Initialize a state space mixture distribution *)
let init_state_space_mixture ~n_components ~n_time_points ~latent_dim ~global_dim ~use_sparse_cholesky =
  let n_latent = n_time_points * latent_dim in
  
  (* Initialize components *)
  let components = Array.init n_components (fun _ ->
    init_state_space_component ~n_time_points ~latent_dim ~global_dim ~use_sparse_cholesky
  ) in
  
  (* Initialize weights with uniform distribution *)
  let weights = Tensor.ones [n_components] |> 
                Tensor.div_scalar (Tensor.of_float (float_of_int n_components)) in
  
  { components; weights; n_components; n_global = global_dim; 
    n_latent; n_subjects = n_time_points; latent_dim }

(* Initialize a new mixture component with various strategies *)
let initialize_component ~base_component ~n_latent ~n_global ~strategy =
  let mean_scale = 0.05 in
  let chol_scale = 0.01 in
  
  let new_mu = initialize_mean 
                ~base_mean:base_component.mu 
                ~n_latent 
                ~n_global 
                ~strategy:(match strategy with
                          | `Random -> `Random
                          | `KMeans data -> `KMeans ~data ~k:5
                          | `Factor -> `Factor
                          | `StateSpace -> `Random)
                ~scale:mean_scale in
  
  let new_l_chol = initialize_cholesky
                    ~base_chol:base_component.l_chol
                    ~n_latent
                    ~n_global
                    ~strategy:(match strategy with
                              | `Random -> `Perturb
                              | `KMeans _ -> `Identity
                              | `Factor -> `Diagonal 1.0
                              | `StateSpace -> `StateSpace)
                    ~scale:chol_scale in
  
  (* Extract sub-matrices *)
  let mu_g = Tensor.narrow new_mu ~dim:0 ~start:n_latent ~length:n_global in
  let mu_b = Tensor.narrow new_mu ~dim:0 ~start:0 ~length:n_latent in
  
  let l_g = Tensor.narrow new_l_chol ~dim:0 ~start:n_latent ~length:n_global |>
           fun x -> Tensor.narrow x ~dim:1 ~start:n_latent ~length:n_global in
  
  (* For a standard model, extract l_b and l_gb *)
  let is_state_space = match strategy with
                      | `StateSpace -> true
                      | _ -> false in
  
  if not is_state_space then begin
    (* Standard model with independent latent variables *)
    let n_subjects = n_latent / (List.length base_component.l_b) in
    let latent_dim = n_latent / n_subjects in
    
    (* Extract Cholesky factors for each latent variable block *)
    let l_b = List.init n_subjects (fun i ->
      let start = i * latent_dim in
      Tensor.narrow new_l_chol ~dim:0 ~start ~length:latent_dim |>
      fun x -> Tensor.narrow x ~dim:1 ~start ~length:latent_dim
    ) in
    
    (* Extract cross-covariance terms *)
    let l_gb = List.init n_subjects (fun i ->
      let start = i * latent_dim in
      Tensor.narrow new_l_chol ~dim:0 ~start:n_latent ~length:n_global |>
      fun x -> Tensor.narrow x ~dim:1 ~start ~length:latent_dim
    ) in
    
    { mu = new_mu; l_chol = new_l_chol; mu_g; mu_b; l_g; l_b; l_gb; l_tilde = None }
  end
  else begin
    (* State space model with time dependencies *)
    let n_time_points = n_latent in
    let latent_dim = 1; (* Assuming 1D state in this implementation *)
    
    (* Extract Cholesky factors for each state *)
    let l_b = List.init n_time_points (fun i ->
      let start = i * latent_dim in
      Tensor.narrow new_l_chol ~dim:0 ~start ~length:latent_dim |>
      fun x -> Tensor.narrow x ~dim:1 ~start ~length:latent_dim
    ) in
    
    (* Extract transition matrices between states *)
    let l_tilde = List.init (n_time_points - 1) (fun i ->
      let curr_start = (i + 1) * latent_dim in
      let prev_start = i * latent_dim in
      Tensor.narrow new_l_chol ~dim:0 ~start:curr_start ~length:latent_dim |>
      fun x -> Tensor.narrow x ~dim:1 ~start:prev_start ~length:latent_dim
    ) in
    
    (* Extract cross-covariance terms *)
    let l_gb = List.init n_time_points (fun i ->
      let start = i * latent_dim in
      Tensor.narrow new_l_chol ~dim:0 ~start:n_latent ~length:n_global |>
      fun x -> Tensor.narrow x ~dim:1 ~start ~length:latent_dim
    ) in
    
    { mu = new_mu; l_chol = new_l_chol; mu_g; mu_b; l_g; l_b; l_gb; l_tilde = Some l_tilde }
  end

(* Calculate log-likelihood of a sample under a Gaussian component *)
let log_prob_gaussian_component component x =
  let n = Tensor.shape x |> List.hd in
  let mu = component.mu in
  let l_chol = component.l_chol in
  
  (* Centered data *)
  let centered = Tensor.(-) x mu in
  
  (* Calculate log determinant from Cholesky factor *)
  let log_det = log_det_from_cholesky l_chol in
  
  (* Calculate quadratic form (x-μ)ᵀΩ(x-μ) using the Cholesky factor *)
  (* First solve the system L'y = x-μ *)
  let y = Tensor.triangular_solve l_chol (Tensor.reshape centered [n; 1]) 
          ~upper:true ~transpose:false ~unitriangular:false in
  
  let quad_form = Tensor.sum (Tensor.mul y y) [0; 1] false in
  
  (* Log probability: -0.5*n*log(2π) + 0.5*log|Ω| - 0.5*(x-μ)ᵀΩ(x-μ) *)
  let log_2pi = Tensor.of_float (float_of_int n *. log (2.0 *. Float.pi)) in
  Tensor.(-)
    (Tensor.(+) 
      (Tensor.mul_scalar (Tensor.neg log_2pi) (Tensor.of_float 0.5))
      (Tensor.mul_scalar log_det (Tensor.of_float 0.5)))
    (Tensor.mul_scalar quad_form (Tensor.of_float 0.5))

(* Calculate log-likelihood of a sample under the Gaussian mixture *)
let log_prob_gaussian_mixture mixture x =
  (* Calculate log probabilities for each component *)
  let log_probs = Array.mapi (fun i component ->
    let log_weight = Tensor.get mixture.weights [i] |> Tensor.log in
    let log_comp_prob = log_prob_gaussian_component component x in
    Tensor.(+) log_weight log_comp_prob
  ) mixture.components in
  
  (* Combine log probabilities using log-sum-exp trick *)
  let log_probs_tensor = Tensor.stack (Array.to_list log_probs) ~dim:0 in
  log_sum_exp log_probs_tensor

(* Get conditional distribution for latent variables given global parameters *)
let conditional_latent_given_global mixture global_params =
  (* Extract global parameters *)
  let n_global = mixture.n_global in
  
  (* Return conditional distribution parameters for each component *)
  Array.mapi (fun i component ->
    (* Calculate conditional means for latent variables *)
    let mu_g = component.mu_g in
    let mu_b = component.mu_b in
    
    (* Calculate centered global params *)
    let centered_global = Tensor.(-) global_params mu_g in
    
    (* Calculate conditional means for each latent block *)
    let cond_means = List.mapi (fun j l_b ->
      let l_gb = List.nth component.l_gb j in
      let cross_term = Tensor.matmul l_gb (Tensor.reshape centered_global [n_global; 1]) in
      let latent_dim = Tensor.shape l_b |> List.hd in
      let start_idx = j * latent_dim in
      let mu_b_j = Tensor.narrow mu_b ~dim:0 ~start:start_idx ~length:latent_dim in
      Tensor.(+) (Tensor.reshape mu_b_j [latent_dim; 1]) (Tensor.matmul l_b cross_term)
    ) component.l_b in
    
    (* Calculate conditional precisions (they are just the diagonal blocks) *)
    let cond_precs = component.l_b in
    
    (mixture.weights |> Tensor.get [i], cond_means, cond_precs)
  ) mixture.components

(* Get marginal distribution for global parameters *)
let marginal_global_distribution mixture =
  (* Return marginal distribution parameters for each component *)
  Array.mapi (fun i component ->
    let weight = Tensor.get mixture.weights [i] in
    let mu_g = component.mu_g in
    let prec_g = Tensor.matmul 
                  (Tensor.transpose component.l_g ~dim0:0 ~dim1:1) 
                  component.l_g in
    (weight, mu_g, prec_g)
  ) mixture.components

(* Get conditional distribution for state n given state n+1 and global params *)
let conditional_state_given_next_and_global component state_idx global_params =
  (* Extract dimensions *)
  let state_dim = Tensor.shape (List.nth component.l_b state_idx) |> List.hd in
  let n_states = List.length component.l_b in
  let n_global = Tensor.shape component.mu_g |> List.hd in
  
  (* Cannot condition on next state if this is the last state *)
  if state_idx >= n_states - 1 then
    failwith "Cannot condition on next state for the last time point";
  
  (* For state space models, we need the transition matrix l_tilde *)
  match component.l_tilde with
  | None -> failwith "Not a state space model component"
  | Some l_tilde ->
      let l_tilde_curr = List.nth l_tilde state_idx in
      
      (* Get parameters *)
      let start_idx = state_idx * state_dim in
      let next_idx = (state_idx + 1) * state_dim in
      
      let mu_curr = Tensor.narrow component.mu_b ~dim:0 ~start:start_idx ~length:state_dim in
      let mu_next = Tensor.narrow component.mu_b ~dim:0 ~start:next_idx ~length:state_dim in
      
      let l_curr = List.nth component.l_b state_idx in
      let l_next = List.nth component.l_b (state_idx + 1) in
      
      (* Calculate centered next state and global params *)
      let next_state = Tensor.narrow component.mu_b ~dim:0 ~start:next_idx ~length:state_dim in
      let centered_next = Tensor.(-) next_state mu_next in
      let centered_global = Tensor.(-) global_params component.mu_g in
      
      (* Calculate cross-covariance terms *)
      let l_gb_curr = List.nth component.l_gb state_idx in
      let l_gb_next = List.nth component.l_gb (state_idx + 1) in
      
      (* Calculate conditional mean: μ_curr + L_tilde·L_next^(-1)·(s_next - μ_next) + L_gb_curr·(θ_G - μ_G) *)
      let l_next_inv_centered = Tensor.triangular_solve l_next (Tensor.reshape centered_next [state_dim; 1])
                              ~upper:true ~transpose:false ~unitriangular:false in
      
      let trans_term = Tensor.matmul l_tilde_curr l_next_inv_centered in
      let global_term = Tensor.matmul l_gb_curr (Tensor.reshape centered_global [n_global; 1]) in
      
      let cond_mean = Tensor.(+) (Tensor.(+) (Tensor.reshape mu_curr [state_dim; 1]) trans_term) global_term in
      
      (* Conditional precision is just L_curr·L_curr^T *)
      let cond_prec = Tensor.matmul (Tensor.transpose l_curr ~dim0:0 ~dim1:1) l_curr in
      
      (Tensor.reshape cond_mean [state_dim], cond_prec)

(* Sample from the Gaussian mixture *)
let sample_gaussian_mixture mixture ~n_samples =
  let total_dim = mixture.n_global + mixture.n_latent in
  let samples = Tensor.zeros [n_samples; total_dim] in
  
  (* Sample component indices based on weights *)
  let weights_cpu = Tensor.to_device mixture.weights ~device:Cpu in
  let weights_array = Array.init mixture.n_components (fun i -> 
    Tensor.get weights_cpu [i] |> Tensor.to_float0_exn) in
  
  for i = 0 to n_samples - 1 do
    (* Sample component index *)
    let u = Random.float 1.0 in
    let mut_sum = ref 0.0 in
    let mut_idx = ref 0 in
    
    for j = 0 to mixture.n_components - 1 do
      mut_sum := !mut_sum +. weights_array.(j);
      if u <= !mut_sum && j >= !mut_idx then
        mut_idx := j
    done;
    
    let idx = !mut_idx in
    let component = mixture.components.(idx) in
    
    (* Sample from selected Gaussian component *)
    let z = Tensor.randn [total_dim] in
    
    (* Transform to get a sample: x = μ + L⁻¹z *)
    let l_inv_z = Tensor.triangular_solve component.l_chol (Tensor.reshape z [total_dim; 1])
                 ~upper:true ~transpose:false ~unitriangular:false in
    let sample = Tensor.(+) component.mu (Tensor.reshape l_inv_z [total_dim]) in
    
    (* Store the sample *)
    for j = 0 to total_dim - 1 do
      Tensor.set samples [i; j] (Tensor.get sample [j])
    done
  done;
  
  samples

(* Calculate ELBO (Evidence Lower Bound) for variational inference *)
let calculate_elbo mixture log_joint_prob =
  (* Sample from the variational distribution *)
  let samples = sample_gaussian_mixture mixture ~n_samples:100 in
  let n_samples = Tensor.shape samples |> List.hd in
  
  (* Calculate log joint probability for each sample *)
  let log_joints = Tensor.zeros [n_samples] in
  for i = 0 to n_samples - 1 do
    let sample = Tensor.select samples ~dim:0 ~index:i in
    let log_joint = log_joint_prob sample in
    Tensor.set log_joints [i] log_joint
  done;
  
  (* Calculate log variational probability for each sample *)
  let log_var_probs = Tensor.zeros [n_samples] in
  for i = 0 to n_samples - 1 do
    let sample = Tensor.select samples ~dim:0 ~index:i in
    let log_var_prob = log_prob_gaussian_mixture mixture sample in
    Tensor.set log_var_probs [i] log_var_prob
  done;
  
  (* ELBO = E_q[log p(x, z) - log q(z)] *)
  let elbo_terms = Tensor.(-) log_joints log_var_probs in
  Tensor.mean elbo_terms [0]