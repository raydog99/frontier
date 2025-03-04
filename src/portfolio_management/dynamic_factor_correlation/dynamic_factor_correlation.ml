open Torch

(* Create a square diagonal matrix *)
let diag values =
  let n = Tensor.size values 0 in
  let result = Tensor.zeros [n; n] in
  for i = 0 to n - 1 do
    Tensor.set_2d result i i (Tensor.get_1d values i)
  done;
  result

(* Matrix logarithm for GFT *)
let matrix_log m =
  (* Using eigendecomposition: P * log(D) * P^(-1) *)
  let n = Tensor.size m 0 in
  
  (* Compute eigendecomposition *)
  let eigvals, eigvecs = Tensor.symeig m ~eigenvectors:true in
  
  (* Apply log to eigenvalues *)
  let log_eigvals = Tensor.log eigvals in
  
  (* Reconstruct: P * log(D) * P^T *)
  let log_eigvals_diag = diag log_eigvals in
  let result = Tensor.matmul eigvecs (Tensor.matmul log_eigvals_diag (Tensor.transpose eigvecs ~dim0:0 ~dim1:1)) in
  
  result

(* Matrix exponential for inverse GFT *)
let matrix_exp m =
  (* Using eigendecomposition: P * exp(D) * P^(-1) *)
  let n = Tensor.size m 0 in
  
  (* Compute eigendecomposition *)
  let eigvals, eigvecs = Tensor.symeig m ~eigenvectors:true in
  
  (* Apply exp to eigenvalues *)
  let exp_eigvals = Tensor.exp eigvals in
  
  (* Reconstruct: P * exp(D) * P^T *)
  let exp_eigvals_diag = diag exp_eigvals in
  let result = Tensor.matmul eigvecs (Tensor.matmul exp_eigvals_diag (Tensor.transpose eigvecs ~dim0:0 ~dim1:1)) in
  
  result
  
(* Fisher transformation and its inverse *)
let fisher_transform x = Tensor.atanh x

let inverse_fisher_transform x = Tensor.tanh x

(* Kronecker product of two matrices *)
let kron a b =
  let a_rows = Tensor.size a 0 in
  let a_cols = Tensor.size a 1 in
  let b_rows = Tensor.size b 0 in
  let b_cols = Tensor.size b 1 in
  
  let result = Tensor.zeros [a_rows * b_rows; a_cols * b_cols] in
  
  for i = 0 to a_rows - 1 do
    for j = 0 to a_cols - 1 do
      let a_ij = Tensor.get_2d a i j in
      
      (* Set block at position (i,j) *)
      for k = 0 to b_rows - 1 do
        for l = 0 to b_cols - 1 do
          let b_kl = Tensor.get_2d b k l in
          Tensor.set_2d result (i * b_rows + k) (j * b_cols + l) (a_ij *. b_kl)
        done
      done
    done
  done;
  
  result
  
(* Compute commutation matrix K_n *)
let compute_commutation_matrix n =
  let result = Tensor.zeros [n * n; n * n] in
  
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      (* Set K_n[i*n+j, j*n+i] = 1 *)
      Tensor.set_2d result (i * n + j) (j * n + i) 1.0
    done
  done;
  
  result
  
(* Compute Upsilon matrix for information matrix *)
let compute_upsilon_matrix n df =
  (* For multivariate t with df degrees of freedom *)
  let weight_factor = (df +. float_of_int n) /. ((df +. float_of_int n +. 2.0) *. (df -. 2.0)) in
  
  (* Upsilon matrix has a simple form for MT *)
  let result = Tensor.zeros [n * n; n * n] in
  
  (* Set all elements to the weight factor *)
  for i = 0 to (n * n) - 1 do
    for j = 0 to (n * n) - 1 do
      Tensor.set_2d result i j weight_factor
    done
  done;
  
  result
  
(* Moore-Penrose inverse with Tikhonov regularization *)
let moore_penrose_inverse_tikhonov m lambda =
  let m_t = Tensor.transpose m ~dim0:0 ~dim1:1 in
  let m_mt = Tensor.matmul m m_t in
  let reg = Tensor.mul_scalar (Tensor.eye (Tensor.size m_t 0)) lambda in
  let inv_term = Tensor.add m_mt reg in
  let inv = Tensor.inverse inv_term in
  Tensor.matmul m_t inv

module UnivariateVolatility = struct
  (* EGARCH(1,1) model parameters *)
  type egarch_params = {
    omega: float;
    alpha: float;
    gamma: float;
    beta: float;
  }

  (* EGARCH forecast function *)
  let egarch_forecast params h_prev z_prev =
    let alpha_term = params.alpha *. (Float.abs z_prev -. sqrt (2.0 /. Float.pi)) in
    let gamma_term = params.gamma *. z_prev in
    let beta_term = params.beta *. h_prev in
    params.omega +. alpha_term +. gamma_term +. beta_term
    
  (* Function to standardize returns using EGARCH volatility *)
  let standardize returns volatilities =
    Tensor.div returns volatilities
end

module FactorLoadings = struct
  (* Transform from factor loadings (rho) to tau parameterization *)
  let to_tau rho =
    let rho_norm_sq = Tensor.dot rho rho in
    let rho_norm = Tensor.sqrt rho_norm_sq in
    let scale = Tensor.div (Tensor.atanh rho_norm) rho_norm in
    Tensor.mul rho scale
    
  (* Transform from tau parameterization back to factor loadings (rho) *)
  let from_tau tau =
    let tau_norm_sq = Tensor.dot tau tau in
    let tau_norm = Tensor.sqrt tau_norm_sq in
    let scale = Tensor.div (Tensor.tanh tau_norm) tau_norm in
    Tensor.mul tau scale
    
  (* Compute Jacobian matrix of the transformation *)
  let jacobian tau =
    let tau_norm_sq = Tensor.dot tau tau in
    let tau_norm = Tensor.sqrt tau_norm_sq in
    let rho = from_tau tau in
    let rho_norm_sq = Tensor.dot rho rho in
    let p_tau = Tensor.div (Tensor.outer tau tau) tau_norm_sq in
    let p_perp = Tensor.sub (Tensor.eye (Tensor.size tau 0)) p_tau in
    
    (* J(tau) = sqrt(rho'rho/tau'tau) * P_perp + (1 - rho'rho) * P_tau *)
    let scalar = Tensor.sqrt (Tensor.div rho_norm_sq tau_norm_sq) in
    let term1 = Tensor.mul p_perp scalar in
    let term2 = Tensor.mul p_tau (Tensor.sub (Tensor.ones []) rho_norm_sq) in
    Tensor.add term1 term2
end

module IdiosyncraticCorrelation = struct
  type block_structure = 
    | Unrestricted
    | FullBlock
    | SparseBlock
    | DiagonalBlock
    
  (* Vectorize lower triangle of a matrix *)
  let vecl m =
    let n = Tensor.size m 0 in
    let result = Tensor.zeros [(n * (n - 1)) / 2] in
    
    let idx = ref 0 in
    for i = 1 to n - 1 do
      for j = 0 to i - 1 do
        Tensor.set_1d result !idx (Tensor.get_2d m i j);
        idx := !idx + 1
      done
    done;
    
    result
    
  (* Convert vecl back to a symmetric matrix *)
  let vecl_to_matrix v n =
    let result = Tensor.eye n in
    
    let idx = ref 0 in
    for i = 1 to n - 1 do
      for j = 0 to i - 1 do
        let val_ = Tensor.get_1d v !idx in
        Tensor.set_2d result i j val_;
        Tensor.set_2d result j i val_;  (* Symmetric matrix *)
        idx := !idx + 1
      done
    done;
    
    result

  (* Parameterize the correlation matrix using generalized Fisher transformation *)
  let gft correlation_matrix =
    let log_corr = matrix_log correlation_matrix in
    vecl log_corr
    
  (* Inverse of generalized Fisher transformation *)
  let inverse_gft gamma =
    (* Determine matrix dimension from gamma size *)
    let gamma_size = Tensor.size gamma 0 in
    let n = Float.sqrt (Float.of_int (gamma_size * 2)) |> Float.ceil |> int_of_float in
    
    let log_corr = vecl_to_matrix gamma n in
    matrix_exp log_corr
    
  (* Create a correlation matrix with equicorrelation block structure *)
  let create_equicorrelation_block n correlation =
    let result = Tensor.ones [n; n] in
    Tensor.mul_scalar_ result correlation;
    for i = 0 to n - 1 do
      Tensor.set_2d result i i 1.0
    done;
    result
    
  (* Create a full block correlation matrix based on group assignments *)
  let create_full_block_matrix group_sizes correlations =
    let n = Array.fold_left (+) 0 group_sizes in
    let result = Tensor.zeros [n; n] in
    
    let current_idx = ref 0 in
    
    (* Fill in the diagonal blocks *)
    for k = 0 to (Array.length group_sizes) - 1 do
      let nk = group_sizes.(k) in
      let start_idx = !current_idx in
      let end_idx = start_idx + nk - 1 in
      
      (* Set diagonal elements to 1 *)
      for i = start_idx to end_idx do
        Tensor.set_2d result i i 1.0
      done;
      
      (* Set off-diagonal elements within the block *)
      for i = start_idx to end_idx do
        for j = start_idx to end_idx do
          if i <> j then
            Tensor.set_2d result i j correlations.(k)
        done
      done;
      
      current_idx := !current_idx + nk
    done;
    
    (* Fill in the off-diagonal blocks *)
    current_idx := 0;
    for k = 0 to (Array.length group_sizes) - 1 do
      let nk = group_sizes.(k) in
      let start_k = !current_idx in
      let end_k = start_k + nk - 1 in
      
      let next_idx = ref (start_k + nk) in
      for l = k + 1 to (Array.length group_sizes) - 1 do
        let nl = group_sizes.(l) in
        let start_l = !next_idx in
        let end_l = start_l + nl - 1 in
        
        (* Get the correlation between blocks k and l *)
        let corr_kl = correlations.(k + l * (Array.length group_sizes)) in
        
        (* Set the off-diagonal block *)
        for i = start_k to end_k do
          for j = start_l to end_l do
            Tensor.set_2d result i j corr_kl;
            Tensor.set_2d result j i corr_kl; (* symmetric matrix *)
          done
        done;
        
        next_idx := !next_idx + nl
      done;
      
      current_idx := !current_idx + nk
    done;
    
    result
    
  (* Create a sparse block correlation matrix (zeros between sectors) *)
  let create_sparse_block_matrix group_sizes sector_assignments correlations =
    let n = Array.fold_left (+) 0 group_sizes in
    let result = Tensor.zeros [n; n] in
    
    (* First create a full block matrix *)
    let full_block = create_full_block_matrix group_sizes correlations in
    
    (* Zero out correlations between different sectors *)
    let current_idx = ref 0 in
    for k = 0 to (Array.length group_sizes) - 1 do
      let nk = group_sizes.(k) in
      let start_k = !current_idx in
      let end_k = start_k + nk - 1 in
      let sector_k = sector_assignments.(k) in
      
      let next_idx = ref (start_k + nk) in
      for l = k + 1 to (Array.length group_sizes) - 1 do
        let nl = group_sizes.(l) in
        let start_l = !next_idx in
        let end_l = start_l + nl - 1 in
        let sector_l = sector_assignments.(l) in
        
        (* If sectors are different, set correlation to zero *)
        if sector_k <> sector_l then
          for i = start_k to end_k do
            for j = start_l to end_l do
              Tensor.set_2d result i j 0.0;
              Tensor.set_2d result j i 0.0;
            done
          done;
        
        next_idx := !next_idx + nl
      done;
      
      current_idx := !current_idx + nk
    done;
    
    result
    
  (* Create a diagonal block correlation matrix (only within-block correlations) *)
  let create_diagonal_block_matrix group_sizes correlations =
    let n = Array.fold_left (+) 0 group_sizes in
    let result = Tensor.zeros [n; n] in
    
    let current_idx = ref 0 in
    for k = 0 to (Array.length group_sizes) - 1 do
      let nk = group_sizes.(k) in
      let start_idx = !current_idx in
      let end_idx = start_idx + nk - 1 in
      
      (* Set diagonal elements to 1 *)
      for i = start_idx to end_idx do
        Tensor.set_2d result i i 1.0
      done;
      
      (* Set off-diagonal elements within the block *)
      for i = start_idx to end_idx do
        for j = start_idx to end_idx do
          if i <> j then
            Tensor.set_2d result i j correlations.(k)
        done
      done;
      
      current_idx := !current_idx + nk
    done;
    
    result
    
  (* Convert eta parameters to correlation coefficient for diagonal block structure *)
  let eta_to_equicorrelation eta nk =
    let eta_nk = Float.of_int nk *. eta in
    let num = Float.exp eta_nk -. 1.0 in
    let denom = Float.exp eta_nk +. Float.of_int nk -. 1.0 in
    num /. denom
end

module CanonicalRepresentation = struct
  (* Create a block indicator matrix for a given group structure *)
  let create_group_indicator_matrix group_sizes =
    let n = Array.fold_left (+) 0 group_sizes in
    let num_groups = Array.length group_sizes in
    
    (* Initialize Q matrix *)
    let q = Tensor.zeros [n; n] in
    
    (* Set blocks in Q matrix *)
    let current_idx = ref 0 in
    for k = 0 to num_groups - 1 do
      let n_k = group_sizes.(k) in
      let start_idx = !current_idx in
      let end_idx = start_idx + n_k - 1 in
      
      (* Set diagonal block to 1/sqrt(n_k) *)
      let block_val = 1.0 /. sqrt (float_of_int n_k) in
      for i = start_idx to end_idx do
        for j = start_idx to end_idx do
          Tensor.set_2d q i j block_val
        done
      done;
      
      current_idx := !current_idx + n_k
    done;
    
    q

  (* Calculate the A matrix in the canonical representation *)
  let calculate_a_matrix group_sizes correlations =
    let num_groups = Array.length group_sizes in
    let a = Tensor.zeros [num_groups; num_groups] in
    
    (* Set diagonal elements *)
    for k = 0 to num_groups - 1 do
      let n_k = group_sizes.(k) in
      let rho_kk = correlations.(k) in
      let a_kk = 1.0 +. (float_of_int n_k -. 1.0) *. rho_kk in
      Tensor.set_2d a k k a_kk
    done;
    
    (* Set off-diagonal elements *)
    for k = 0 to num_groups - 1 do
      let n_k = group_sizes.(k) in
      for l = k+1 to num_groups - 1 do
        let n_l = group_sizes.(l) in
        let idx = num_groups * k + l in
        let rho_kl = correlations.(idx) in
        let a_kl = rho_kl *. sqrt (float_of_int (n_k * n_l)) in
        Tensor.set_2d a k l a_kl;
        Tensor.set_2d a l k a_kl; (* Symmetric matrix *)
      done
    done;
    
    a

  (* Calculate the delta values in the canonical representation *)
  let calculate_deltas group_sizes a_matrix =
    let num_groups = Array.length group_sizes in
    let deltas = Array.make num_groups 0.0 in
    
    for k = 0 to num_groups - 1 do
      let n_k = group_sizes.(k) in
      let a_kk = Tensor.get_2d a_matrix k k in
      deltas.(k) <- (float_of_int n_k -. a_kk) /. (float_of_int n_k -. 1.0)
    done;
    
    deltas

  (* Compute canonical representation of a block correlation matrix *)
  let compute_canonical_representation group_sizes correlations =
    let q = create_group_indicator_matrix group_sizes in
    let a = calculate_a_matrix group_sizes correlations in
    let deltas = calculate_deltas group_sizes a in
    
    (q, a, deltas)

  (* Compute correlation matrix from canonical form *)
  let compute_correlation_from_canonical q a deltas group_sizes =
    let num_groups = Array.length group_sizes in
    let n = Tensor.size q 0 in
    
    (* Create D matrix *)
    let d = Tensor.zeros [n; n] in
    
    (* Set top-left block to A *)
    for i = 0 to num_groups - 1 do
      for j = 0 to num_groups - 1 do
        Tensor.set_2d d i j (Tensor.get_2d a i j)
      done
    done;
    
    (* Set diagonal blocks for eigenvalues *)
    let current_idx = ref num_groups in
    for k = 0 to num_groups - 1 do
      let n_k = group_sizes.(k) in
      for _ = 1 to n_k - 1 do
        Tensor.set_2d d !current_idx !current_idx deltas.(k);
        current_idx := !current_idx + 1
      done
    done;
    
    (* Compute C = Q D Q' *)
    let q_t = Tensor.transpose q ~dim0:0 ~dim1:1 in
    let qd = Tensor.matmul q d in
    let c = Tensor.matmul qd q_t in
    
    c

  (* Convert correlation coefficients to eta parameters *)
  let correlation_to_eta group_sizes correlations =
    let num_groups = Array.length group_sizes in
    let _, a, _ = compute_canonical_representation group_sizes correlations in
    
    (* η = LK(Λ_n^-1 ⊗ Λ_n^-1)vec(W) *)
    let eta = Array.make num_groups 0.0 in
    
    for k = 0 to num_groups - 1 do
      let n_k = group_sizes.(k) in
      let rho_kk = correlations.(k) in
      
      eta.(k) <- 1.0 /. (float_of_int n_k) *. 
                 log (1.0 +. (float_of_int n_k) *. rho_kk /. (1.0 -. rho_kk))
    done;
    
    eta

  (* Convert eta parameters to correlation coefficients *)
  let eta_to_correlation group_sizes eta =
    let num_groups = Array.length group_sizes in
    let correlations = Array.make (num_groups * (num_groups + 1) / 2) 0.0 in
    
    for k = 0 to num_groups - 1 do
      let n_k = group_sizes.(k) in
      let eta_k = eta.(k) in
      
      let exp_term = exp ((float_of_int n_k) *. eta_k) in
      correlations.(k) <- (exp_term -. 1.0) /. (exp_term +. (float_of_int n_k) -. 1.0)
    done;
    
    correlations
end

module Distributions = struct
  type convolution_t_type =
    | MultivariateTDist
    | ClusterTDist
    | HeterogeneousTDist

  (* Log-likelihood of standardized t-distribution *)
  let log_likelihood_t_dist df x =
    let n = Tensor.size x 0 in
    let df_float = float_of_int df in
    
    (* Compute normalizing constant *)
    let ln_gamma_term = Stdlib.log (Stdlib.gamma ((df_float +. float_of_int n) /. 2.0)) -. 
                         Stdlib.log (Stdlib.gamma (df_float /. 2.0)) in
    let c = ln_gamma_term -. (float_of_int n /. 2.0) *. Stdlib.log ((df_float -. 2.0) *. Float.pi) in
    
    (* Compute quadratic form *)
    let quad_form = Tensor.dot x x in
    
    (* Compute log-likelihood *)
    let ll = c -. ((df_float +. float_of_int n) /. 2.0) *. 
             Stdlib.log (1.0 +. (Tensor.to_float1 quad_form) /. (df_float -. 2.0)) in
    
    ll
    
  (* Log-likelihood for Multivariate-t distribution *)
  let log_likelihood_multivariate_t df mu sigma x =
    let centered = Tensor.sub x mu in
    let precision = Tensor.inverse sigma in
    let normalized = Tensor.matmul centered (Tensor.matmul precision (Tensor.unsqueeze centered 1)) in
    let normalized = Tensor.squeeze normalized ~dim:1 in
    
    let n = Tensor.size x 0 in
    let df_float = float_of_int df in
    
    (* Compute normalizing constant *)
    let ln_gamma_term = Stdlib.log (Stdlib.gamma ((df_float +. float_of_int n) /. 2.0)) -. 
                         Stdlib.log (Stdlib.gamma (df_float /. 2.0)) in
    let c = ln_gamma_term -. (float_of_int n /. 2.0) *. Stdlib.log ((df_float -. 2.0) *. Float.pi) -. 
            0.5 *. Stdlib.log (Tensor.to_float0 (Tensor.det sigma)) in
    
    (* Compute log-likelihood *)
    let ll = c -. ((df_float +. float_of_int n) /. 2.0) *. 
             Stdlib.log (1.0 +. (Tensor.to_float0 normalized) /. (df_float -. 2.0)) in
    
    ll
    
  (* Log-likelihood components for convolution-t distribution *)
  let log_likelihood_components_convolution_t mu sigma v distribution_type dfs group_sizes =
    match distribution_type with
    | MultivariateTDist ->
        (* For MT, one log-likelihood value *)
        let n = Tensor.size mu 0 in
        let df = Array.get dfs 0 in  (* Using first df for MT *)
        let df_float = float_of_int df in
        
        (* Compute normalizing constant *)
        let ln_gamma_term = Stdlib.log (Stdlib.gamma ((df_float +. float_of_int n) /. 2.0)) -. 
                           Stdlib.log (Stdlib.gamma (df_float /. 2.0)) in
        let c = ln_gamma_term -. (float_of_int n /. 2.0) *. Stdlib.log ((df_float -. 2.0) *. Float.pi) in
        
        (* Precision matrix *)
        let precision = Tensor.inverse sigma in
        
        (* Mahalanobis distance *)
        let centered = Tensor.sub v mu in
        let quad_form = Tensor.matmul (Tensor.unsqueeze centered 0) 
                                     (Tensor.matmul precision (Tensor.unsqueeze centered 1)) in
        let quad_form = Tensor.squeeze quad_form in
        
        (* Log determinant term *)
        let log_det = Tensor.logdet sigma in
        
        (* Complete log-likelihood *)
        let ll = c -. 0.5 *. Tensor.to_float0 log_det -. 
                 ((df_float +. float_of_int n) /. 2.0) *. 
                 Stdlib.log (1.0 +. (Tensor.to_float0 quad_form) /. (df_float -. 2.0)) in
        
        [| ll |]
        
    | ClusterTDist ->
        (* For CT, separate log-likelihood for each cluster *)
        let num_groups = Array.length group_sizes in
        let ll_values = Array.make num_groups 0.0 in
        
        let group_start = ref 0 in
        for g = 0 to num_groups - 1 do
          let group_size = group_sizes.(g) in
          let group_end = !group_start + group_size in
          
          (* Extract group components *)
          let group_mu = Tensor.narrow mu ~dim:0 ~start:!group_start ~length:group_size in
          
          (* Extract submatrix from sigma *)
          let group_sigma = Tensor.zeros [group_size; group_size] in
          for i = 0 to group_size - 1 do
            for j = 0 to group_size - 1 do
              let val_ = Tensor.get_2d sigma (!group_start + i) (!group_start + j) in
              Tensor.set_2d group_sigma i j val_
            done
          done;
          
          let group_v = Tensor.narrow v ~dim:0 ~start:!group_start ~length:group_size in
          
          (* Get appropriate df for this group *)
          let df = if g < Array.length dfs then dfs.(g) else dfs.(0) in
          let df_float = float_of_int df in
          
          (* Compute normalizing constant *)
          let ln_gamma_term = Stdlib.log (Stdlib.gamma ((df_float +. float_of_int group_size) /. 2.0)) -. 
                            Stdlib.log (Stdlib.gamma (df_float /. 2.0)) in
          let c = ln_gamma_term -. (float_of_int group_size /. 2.0) *. 
                  Stdlib.log ((df_float -. 2.0) *. Float.pi) in
          
          (* Precision matrix *)
          let precision = Tensor.inverse group_sigma in
          
          (* Mahalanobis distance *)
          let centered = Tensor.sub group_v group_mu in
          let quad_form = Tensor.matmul (Tensor.unsqueeze centered 0) 
                                      (Tensor.matmul precision (Tensor.unsqueeze centered 1)) in
          let quad_form = Tensor.squeeze quad_form in
          
          (* Log determinant term *)
          let log_det = Tensor.logdet group_sigma in
          
          (* Complete log-likelihood for this group *)
          let ll = c -. 0.5 *. Tensor.to_float0 log_det -. 
                  ((df_float +. float_of_int group_size) /. 2.0) *. 
                  Stdlib.log (1.0 +. (Tensor.to_float0 quad_form) /. (df_float -. 2.0)) in
          
          ll_values.(g) <- ll;
          group_start := group_end
        done;
        
        ll_values
        
    | HeterogeneousTDist ->
        (* For HT, separate log-likelihood for each component *)
        let n = Tensor.size mu 0 in
        let ll_values = Array.make n 0.0 in
        
        for i = 0 to n - 1 do
          (* Get component values *)
          let mu_i = Tensor.get_1d mu i in
          let sigma_i = Tensor.get_2d sigma i i |> sqrt in
          let v_i = Tensor.get_1d v i in
          
          (* Get appropriate df *)
          let df = if i < Array.length dfs then dfs.(i) else dfs.(0) in
          let df_float = float_of_int df in
          
          (* Compute normalizing constant *)
          let ln_gamma_term = Stdlib.log (Stdlib.gamma ((df_float +. 1.0) /. 2.0)) -. 
                            Stdlib.log (Stdlib.gamma (df_float /. 2.0)) in
          let c = ln_gamma_term -. 0.5 *. Stdlib.log ((df_float -. 2.0) *. Float.pi) in
          
          (* Standardized value *)
          let z_i = (v_i -. mu_i) /. sigma_i in
          
          (* Log-likelihood for this component *)
          let ll = c -. Stdlib.log sigma_i -. 
                  ((df_float +. 1.0) /. 2.0) *. 
                  Stdlib.log (1.0 +. (z_i *. z_i) /. (df_float -. 2.0)) in
          
          ll_values.(i) <- ll
        done;
        
        ll_values

  (* Compute information matrix for MT *)
  let compute_mt_information_matrix mu sigma df =
    let n = Tensor.size mu 0 in
    let df_float = float_of_int df in
    
    (* Compute term for mean *)
    let i_mu = Tensor.div_scalar 
      (Tensor.inverse sigma)
      ((df_float +. float_of_int n) /. ((df_float +. float_of_int n +. 2.0) *. (df_float -. 2.0))) in
    
    let precision = Tensor.inverse sigma in
    let term1 = Tensor.kron precision (Tensor.eye n) in
    let term2 = Tensor.add k_n upsilon in
    let term3 = Tensor.kron (Tensor.eye n) precision in
    
    let i_sigma = Tensor.matmul term1 (Tensor.matmul term2 term3) in
    
    (i_mu, i_sigma)


module ScoreDriven = struct
  (* Core parameters for the score-driven model *)
  type score_driven_params = {
    kappa: Tensor.t;  (* Intercept *)
    beta: Tensor.t;   (* Persistence *)
    alpha: Tensor.t;  (* Score coefficient *)
  }
  
  (* One step of score-driven update *)
  let update params current_value score =
    let intercept_term = Tensor.mul (Tensor.sub (Tensor.ones []) params.beta) params.kappa in
    let persistence_term = Tensor.mul params.beta current_value in
    let score_term = Tensor.mul params.alpha score in
    Tensor.add intercept_term (Tensor.add persistence_term score_term)
    
  (* Compute score for the factor loadings update *)  
  let compute_factor_loading_score tau u z omega df =
    let rho = FactorLoadings.from_tau tau in
    let j = FactorLoadings.jacobian tau in
    
    (* Compute expected value *)
    let mu = Tensor.dot rho u in
    
    (* Compute error *)
    let e = (z -. mu) /. omega in
    
    (* Compute weight W *)
    let df_float = float_of_int df in
    let weight = (df_float +. 1.0) /. (df_float -. 2.0 +. (e *. e)) in
    
    (* Compute factor loading score components *)
    let score_term1 = Tensor.mul_scalar u (weight *. e /. omega) in
    let score_term2 = Tensor.mul_scalar rho ((1.0 -. weight *. e *. e) /. (omega *. omega)) in
    
    (* Final score for tau *)
    let score = Tensor.add score_term1 score_term2 in
    Tensor.matmul j score
    
  (* Compute score for the idiosyncratic correlation update *)
  let compute_idiosyncratic_corr_score eta e dist_type df group_sizes =
    match dist_type with
    | Distributions.MultivariateTDist ->
        (* Multivariate t distribution *)
        let n = Tensor.size e 0 in
        let df_float = float_of_int df in
        
        (* Compute C_e from eta *)
        let c_e = IdiosyncraticCorrelation.inverse_gft eta in
        
        (* Compute quadratic form e^T C_e^-1 e *)
        let c_e_inv = Tensor.inverse c_e in
        let e_unsqueezed = Tensor.unsqueeze e 0 in
        let quad_form = Tensor.matmul e_unsqueezed (Tensor.matmul c_e_inv (Tensor.transpose e_unsqueezed ~dim0:0 ~dim1:1)) in
        let quad_form = Tensor.squeeze quad_form in
        
        (* Compute weight W *)
        let weight = (df_float +. float_of_int n) /. (df_float -. 2.0 +. Tensor.to_float0 quad_form) in
        
        (* Compute vec(Λ) where Λ = C_e^-1 e e^T C_e^-1 - C_e^-1 *)
        let e_e_t = Tensor.outer e e in
        let term1 = Tensor.matmul c_e_inv (Tensor.matmul e_e_t c_e_inv) in
        let lambda = Tensor.sub term1 c_e_inv in
        
        (* Apply vectorization and selection for lower triangle *)
        let vec_lambda = IdiosyncraticCorrelation.vecl lambda in
        
        (* Apply weight and compute final score *)
        Tensor.mul_scalar vec_lambda (weight /. 2.0)
        
    | Distributions.HeterogeneousTDist ->
        let num_groups = Array.length group_sizes in
        let scores = Array.make num_groups (Tensor.zeros []) in
        
        let idx_offset = ref 0 in
        for k = 0 to num_groups - 1 do
          let n_k = group_sizes.(k) in
          let e_k = Tensor.narrow e ~dim:0 ~start:!idx_offset ~length:n_k in
          
          (* Convert eta_k to correlation *)
          let rho_kk = IdiosyncraticCorrelation.eta_to_equicorrelation (Tensor.to_float0 eta_k) n_k in
          
          (* Compute score for this block *)
          let j = 1.0 /. ((1.0 -. rho_kk) *. (1.0 +. (float_of_int n_k -. 1.0) *. rho_kk)) in
          
          (* Compute score terms *)
          let term1 = -0.5 *. j *. ((float_of_int n_k -. 1.0) /. (1.0 +. (float_of_int n_k -. 1.0) *. rho_kk) -. 
                                   (float_of_int n_k -. 1.0) /. (1.0 -. rho_kk)) in
                                   
          (* Compute X_i - ι_n^T X / n terms *)
          let mean_e_k = Tensor.mean e_k ~dim:[0] in
          let centered_e_k = Tensor.sub e_k (Tensor.ones [n_k] |> Tensor.mul_scalar mean_e_k) in
          
          (* Calculate Vi terms *)
          let ones_vec = Tensor.ones [n_k] in
          let iota_x = Tensor.dot ones_vec e_k in
          
          (* Vi = (nXi - ι'X) / (n√(1-ρ)) + ι'X / (n√(1+(n-1)ρ)) *)
          let vi_vals = Array.init n_k (fun i ->
            let xi = Tensor.get_1d e_k i in
            let term1 = (xi *. float_of_int n_k -. iota_x) /. (float_of_int n_k *. sqrt (1.0 -. rho_kk)) in
            let term2 = iota_x /. (float_of_int n_k *. sqrt (1.0 +. (float_of_int n_k -. 1.0) *. rho_kk)) in
            term1 +. term2
          ) in
          
          (* Compute weights for each observation *)
          let df_k = Array.get df k in
          let weights = Array.init n_k (fun i ->
            let vi = vi_vals.(i) in
            let df_float = float_of_int df_k in
            (df_float +. 1.0) /. (df_float -. 2.0 +. vi *. vi)
          ) in
          
          (* Compute sum of Wi*Vi terms *)
          let sum_w_v = ref 0.0 in
          for i = 0 to n_k - 1 do
            sum_w_v := !sum_w_v +. weights.(i) *. vi_vals.(i)
          done;
          
          (* Final score term *)
          let term2 = -0.5 *. j *. !sum_w_v in
          
          (* Store score for this block *)
          scores.(k) <- Tensor.ones [1] |> Tensor.mul_scalar (term1 +. term2)
          
          idx_offset := !idx_offset + n_k
        done;
        
        (* Combine scores from all blocks *)
        let combined_score = Tensor.zeros [num_groups] in
        for k = 0 to num_groups - 1 do
          Tensor.set_1d combined_score k (Tensor.get_1d scores.(k) 0)
        done;
        combined_score
    
  (* Tikhonov regularized score *)
  let tikhonov_regularized_score score info_matrix lambda =
    let m_t = Tensor.transpose info_matrix ~dim0:0 ~dim1:1 in
    let m_mt = Tensor.matmul info_matrix m_t in
    let reg = Tensor.mul_scalar (Tensor.eye (Tensor.size m_t 0)) lambda in
    let inv_term = Tensor.add m_mt reg in
    let inv = Tensor.inverse inv_term in
    let mp_inv = Tensor.matmul m_t inv in
    Tensor.matmul mp_inv score
end

module DynamicFactorCorrelation = struct
  (* Full model parameters *)
  type model_params = {
    (* Factor correlation parameters *)
    factor_corr_params: ScoreDriven.score_driven_params;
    
    (* Factor loadings parameters - one set for each asset *)
    factor_loading_params: ScoreDriven.score_driven_params array;
    
    (* Idiosyncratic correlation parameters *)
    idiosyncratic_corr_params: ScoreDriven.score_driven_params;
    
    (* Distribution parameters *)
    factor_dist_type: Distributions.convolution_t_type;
    factor_df: int;
    
    returns_dist_type: Distributions.convolution_t_type;
    returns_df: int array;
    
    (* Regularization parameters *)
    tikhonov_lambda: float array;
  }
  
  (* State variables that evolve over time *)
  type model_state = {
    (* Factor correlation *)
    factor_corr: Tensor.t;
    
    (* Factor loadings for each asset *)
    factor_loadings: Tensor.t array;
    
    (* Idiosyncratic correlation *)
    idiosyncratic_corr: Tensor.t;
  }
  
  (* Initialize a new model state *)
  let init_state n_assets n_factors block_structure =
    let factor_corr = Tensor.eye n_factors in
    
    let factor_loadings = Array.init n_assets (fun _ ->
      Tensor.zeros [n_factors]
    ) in
    
    let idiosyncratic_corr = Tensor.eye n_assets in
    
    { factor_corr; factor_loadings; idiosyncratic_corr }
    
  (* Calculate idiosyncratic residuals *)
  let compute_idiosyncratic_residuals z_t u_t factor_loadings = 
    let n = Array.length factor_loadings in
    let e = Tensor.zeros [n] in
    
    for i = 0 to n - 1 do
      let rho_i = factor_loadings.(i) in
      let mean_i = Tensor.dot rho_i u_t in
      let omega_i = Tensor.sqrt (Tensor.sub (Tensor.ones []) (Tensor.dot rho_i rho_i)) |> Tensor.to_float0 in
      let e_i = (Tensor.get_1d z_t i -. mean_i) /. omega_i in
      Tensor.set_1d e i e_i
    done;
    
    e
  
  (* One step of joint model update - full updating step *)
  let update_joint params state z_t f_t =
    (* Orthogonalize factors *)
    let cf_sqrt = Tensor.cholesky state.factor_corr in
    let cf_inv_sqrt = Tensor.inverse cf_sqrt in
    let u_t = Tensor.matmul cf_inv_sqrt f_t in
    
    (* Convert factor loadings from rho to tau parameterization *)
    let taus = Array.map FactorLoadings.to_tau state.factor_loadings in
    
    (* Calculate omega values *)
    let omegas = Array.map (fun rho ->
      Tensor.sqrt (Tensor.sub (Tensor.ones []) (Tensor.dot rho rho))
    ) state.factor_loadings in
    
    (* Update factor correlation *)
    let new_factor_corr = state.factor_corr in
    
    (* Update factor loadings *)
    let new_factor_loadings = Array.mapi (fun i tau ->
      let score = ScoreDriven.compute_factor_loading_score 
        tau
        u_t
        (Tensor.get_1d z_t i)
        (Tensor.to_float0 (Array.get omegas i))
        params.returns_df.(i) in
      
      let info_matrix = Distributions.compute_factor_loading_info
        tau
        u_t
        params.returns_df.(i) in
        
      let regularized_score = ScoreDriven.tikhonov_regularized_score
        score
        info_matrix
        params.tikhonov_lambda.(i) in
        
      let new_tau = ScoreDriven.update
        params.factor_loading_params.(i)
        tau
        regularized_score in
        
      FactorLoadings.from_tau new_tau
    ) taus in
    
    (* Update idiosyncratic correlation *)
    let e = compute_idiosyncratic_residuals z_t u_t state.factor_loadings in
    
    (* Get block structure sizes *)
    let group_sizes = match params.returns_dist_type with
      | Distributions.MultivariateTDist -> [| Tensor.size z_t 0 |]  (* One big group *)
      | Distributions.ClusterTDist -> failwith "Extract cluster sizes"
      | Distributions.HeterogeneousTDist -> failwith "Extract HT group sizes"
    in
    
    (* Compute score and update *)
    let eta = IdiosyncraticCorrelation.gft state.idiosyncratic_corr in
    let idio_corr_score = ScoreDriven.compute_idiosyncratic_corr_score
      eta
      e
      params.returns_dist_type
      params.returns_df.(0)  (* Using first df as a simplification *)
      group_sizes in
      
    let idio_corr_info = Distributions.compute_idiosyncratic_corr_info
      eta
      params.returns_dist_type
      params.returns_df.(0)
      group_sizes in
      
    let scaled_score = ScoreDriven.tikhonov_regularized_score
      idio_corr_score
      idio_corr_info
      0.001 in  (* Using a small default lambda *)
      
    let new_eta = ScoreDriven.update
      params.idiosyncratic_corr_params
      eta
      scaled_score in
      
    let new_idio_corr = IdiosyncraticCorrelation.inverse_gft new_eta in
    
    { 
      factor_corr = new_factor_corr;
      factor_loadings = new_factor_loadings;
      idiosyncratic_corr = new_idio_corr;
    }
    

        (Array.get omegas i) in
      
      let regularized_score = ScoreDriven.tikhonov_regularized_score
        score
        (failwith "Compute information matrix")
        params.tikhonov_lambda.(i) in
        
      let new_tau = ScoreDriven.update
        params.factor_loading_params.(i)
        tau
        regularized_score in
        
      FactorLoadings.from_tau new_tau
    ) taus in
    
    new_factor_loadings
    
  (* Decoupled estimation - second step: update idiosyncratic correlation *)
  let update_idiosyncratic_corr params state e =
    (* Get block structure sizes *)
    let group_sizes = match params.returns_dist_type with
      | Distributions.MultivariateTDist -> [| Tensor.size e 0 |]  (* One big group *)
      | Distributions.HeterogeneousTDist -> 
          (* For HT, each observation is its own group *)
          Array.init (Tensor.size e 0) (fun _ -> 1)
    in
    
    (* Compute score and information matrix *)
    let eta = IdiosyncraticCorrelation.gft state.idiosyncratic_corr in
    
    let idio_corr_score = ScoreDriven.compute_idiosyncratic_corr_score
      eta
      e
      params.returns_dist_type
      params.returns_df.(0)  (* Using first df as a simplification *)
      group_sizes in
      
    let idio_corr_info = Distributions.compute_idiosyncratic_corr_info
      eta
      params.returns_dist_type
      params.returns_df.(0)
      group_sizes in
      
    let scaled_score = ScoreDriven.tikhonov_regularized_score
      idio_corr_score
      idio_corr_info
      0.001 in  
      
    let new_eta = ScoreDriven.update
      params.idiosyncratic_corr_params
      eta
      scaled_score in
      
    IdiosyncraticCorrelation.inverse_gft new_eta
end