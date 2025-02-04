open Torch

let hadamard_product_n tensors =
  match tensors with
  | [|t|] -> t 
  | tensors -> 
      Array.fold_left (fun acc t ->
        Tensor.mul acc t
      ) tensors.(0) (Array.sub tensors 1 (Array.length tensors - 1))

let kronecker_product a b =
  let a_shape = Tensor.shape a in
  let b_shape = Tensor.shape b in
  let m1, n1 = a_shape.(0), a_shape.(1) in
  let m2, n2 = b_shape.(0), b_shape.(1) in
  
  let result = Tensor.zeros [m1 * m2; n1 * n2] in
  for i = 0 to m1-1 do
    for j = 0 to n1-1 do
      let sub_matrix = Tensor.mul_scalar (Tensor.get a [|i;j|]) b in
      let start_i = i * m2 in
      let start_j = j * n2 in
      Tensor.narrow result ~dim:0 ~start:start_i ~length:m2
      |> fun t -> Tensor.narrow t ~dim:1 ~start:start_j ~length:n2 
      |> fun t -> Tensor.copy_ t sub_matrix
    done
  done;
  result

let khatri_rao_product a b =
  let a_shape = Tensor.shape a in
  let b_shape = Tensor.shape b in
  let m1, r1 = a_shape.(0), a_shape.(1) in
  let m2, _ = b_shape.(0), b_shape.(1) in
  
  let result = Tensor.zeros [m1 * m2; r1] in
  for r = 0 to r1-1 do
    let a_col = Tensor.select a ~dim:1 ~index:r in
    let b_col = Tensor.select b ~dim:1 ~index:r in
    let prod = kronecker_product (Tensor.unsqueeze a_col ~dim:1) 
                               (Tensor.unsqueeze b_col ~dim:1) in
    Tensor.copy_ (Tensor.select result ~dim:1 ~index:r) 
                 (Tensor.squeeze prod ~dim:1)
  done;
  result

let khatri_rao_product_list = function
  | [t] -> t
  | hd::tl ->
      List.fold_left (fun acc t ->
        khatri_rao_product acc t
      ) hd tl
  | _ -> Tensor.zeros [0]

let generalized_inner_product tensors =
  let prod = hadamard_product_n (Array.of_list tensors) in
  Tensor.sum prod

let matricize tensor mode =
  let tensor_shape = Tensor.shape tensor in
  let n_modes = Array.length tensor_shape in
  let mode_dim = tensor_shape.(mode) in
  let other_dims = Array.init n_modes (fun i -> 
    if i <> mode then tensor_shape.(i) else 1
  ) |> Array.fold_left ( * ) 1 in
  
  let perm = Array.init n_modes (fun i ->
    if i = mode then 0
    else if i < mode then i + 1
    else i
  ) in
  
  let permuted = Tensor.permute tensor perm in
  Tensor.reshape permuted [mode_dim; other_dims]

let tensor_product tensors =
  match tensors with
  | [|t|] -> t
  | _ ->
      let flattened = Array.map Tensor.flatten tensors in
      Array.fold_left (fun acc t ->
        Tensor.mul acc t
      ) flattened.(0) (Array.sub flattened 1 (Array.length tensors - 1))
      |> Tensor.sum

let cp_reconstruction factor_matrices =
  let rank = (Tensor.shape factor_matrices.(0)).(1) in
  let dims = Array.map (fun m -> (Tensor.shape m).(0)) factor_matrices in
  let result = Tensor.zeros dims in
  
  for r = 0 to rank - 1 do
    let rank_vectors = Array.map (fun m ->
      Tensor.select m ~dim:1 ~index:r
    ) factor_matrices in
    
    (* Compute outer product *)
    let rank_one = tensor_product rank_vectors in
    Tensor.add_ result rank_one
  done;
  result

let stabilize_precision tensor =
  Tensor.clamp_min tensor eps

let safe_log tensor =
  Tensor.(log (add tensor (full_like tensor eps)))

let safe_div num denom =
  Tensor.(div num (add denom (full_like denom eps)))

let safe_sqrt tensor =
  Tensor.sqrt (Tensor.add tensor (Tensor.full_like tensor eps))

(* Gamma distribution *)
module Gamma = struct
  type t = {
    shape: float;
    rate: float;
  }

  let mean t = t.shape /. t.rate
  let variance t = t.shape /. (t.rate *. t.rate)
  
  let log_expectation t =
    Float.digamma t.shape -. Float.log t.rate

  let entropy t =
    t.shape -. Float.log t.rate +. Float.log_gamma t.shape +.
    (1.0 -. t.shape) *. Float.digamma t.shape

  let kl_divergence q p =
    let term1 = p.shape *. Float.log p.rate -. Float.log_gamma p.shape in
    let term2 = (p.shape -. q.shape) *. Float.digamma q.shape in
    let term3 = q.shape *. p.rate /. q.rate in
    term1 +. term2 +. term3 -. q.shape
end

(* Multivariate normal distribution *)
module MultivariateNormal = struct
  type t = {
    mean: Tensor.t;
    covariance: Tensor.t;
    precision: Tensor.t option;
  }

  let create mean covariance =
    let precision = Some (Tensor.inverse covariance) in
    { mean; covariance; precision }

  let create_with_precision mean precision =
    let covariance = Tensor.inverse precision in
    { mean; covariance; precision = Some precision }

  let log_prob t x =
    let k = Float.of_int (Tensor.shape t.mean |> Array.get 0) in
    let centered = Tensor.sub x t.mean in
    let precision = match t.precision with
      | Some p -> p
      | None -> Tensor.inverse t.covariance in
    
    let quad_term = Tensor.(mm (mm (transpose centered ~dim0:0 ~dim1:1) 
                                  precision) 
                             centered)
                   |> Tensor.get_float1 in
    
    -0.5 *. (k *. Float.log (2.0 *. Float.pi) +. 
             Tensor.logdet t.covariance +. 
             quad_term)

  let kl_divergence q p =
    let k = Float.of_int (Tensor.shape q.mean |> Array.get 0) in
    let p_prec = match p.precision with
      | Some prec -> prec
      | None -> Tensor.inverse p.covariance in
    
    let term1 = Tensor.trace (Tensor.mm p_prec q.covariance) in
    let mean_diff = Tensor.sub q.mean p.mean in
    let term2 = Tensor.(
      mm (mm (transpose mean_diff ~dim0:0 ~dim1:1) p_prec) mean_diff
    ) |> Tensor.get_float1 in
    
    let term3 = Tensor.logdet p.covariance -. Tensor.logdet q.covariance in
    
    0.5 *. (term1 +. term2 -. float k +. term3)
end

(* Model configuration *)
module ModelConfig = struct
  type t = {
    order: int;
    dimensions: int array;
    rank: int;
    noise_precision: float;
    max_iter: int;
    tolerance: float;
  }

  let create order dims rank noise_prec max_iter tol =
    {
      order;
      dimensions = dims;
      rank;
      noise_precision = noise_prec;
      max_iter;
      tolerance = tol;
    }
end

(* Model state *)
module ModelState = struct
  type t = {
    factor_means: Tensor.t array;
    factor_covs: Tensor.t array;
    lambda_shape: float;
    lambda_rate: Tensor.t;
    sparse_mean: Tensor.t;
    sparse_precision: Tensor.t;
    gamma_shape: float;
    gamma_rate: Tensor.t;
    tau_shape: float;
    tau_rate: float;
    elbo: float;
    iteration: int;
  }

  let create config =
    let factor_means = Array.init config.order (fun i ->
      let std = 1.0 /. Float.sqrt (Float.of_int config.rank) in
      Tensor.randn [config.dimensions.(i); config.rank] ~std
    ) in
    
    let factor_covs = Array.init config.order (fun i ->
      Tensor.eye (config.rank)
    ) in
    
    {
      factor_means;
      factor_covs;
      lambda_shape = 1e-6;
      lambda_rate = Tensor.ones [config.rank];
      sparse_mean = Tensor.zeros config.dimensions;
      sparse_precision = Tensor.ones config.dimensions;
      gamma_shape = 1e-6;
      gamma_rate = Tensor.ones config.dimensions;
      tau_shape = 1e-6;
      tau_rate = 1e-6;
      elbo = neg_infinity;
      iteration = 0;
    }
end

(* Posterior update computations *)
module PosteriorUpdates = struct
  (* Update factor matrix posteriors *)
  let update_factor_posterior state tensor mask mode =
    let n_modes = Array.length state.factor_means in
    let current_shape = Tensor.shape state.factor_means.(mode) in
    let n_rows = current_shape.(0) in
    let rank = current_shape.(1) in
    
    let expect_tau = state.tau_shape /. state.tau_rate in
    let expect_lambda = Tensor.div 
      (Tensor.full [rank] state.lambda_shape) 
      state.lambda_rate in
    
    (* Initialize new parameters *)
    let new_mean = Tensor.zeros current_shape in
    let new_cov = Array.make n_rows (Tensor.zeros [rank; rank]) in
    
    (* Get other modes for Khatri-Rao product *)
    let other_modes = Array.init (n_modes - 1) (fun i ->
      if i < mode then state.factor_means.(i)
      else state.factor_means.(i + 1)
    ) in
    
    (* Compute Khatri-Rao expectations *)
    let kr_expect = khatri_rao_expectations 
      other_modes 
      (Array.init (n_modes - 1) (fun i ->
        if i < mode then state.factor_covs.(i)
        else state.factor_covs.(i + 1)
      )) in
    
    (* Update each row *)
    for i = 0 to n_rows - 1 do
      (* Compute posterior precision matrix *)
      let v_inv = Tensor.(
        add 
          (mul_scalar expect_tau kr_expect)
          (diag expect_lambda)
      ) |> stabilize_precision in
      
      (* Compute posterior covariance *)
      let v = Tensor.inverse v_inv in
      new_cov.(i) <- v;
      
      (* Compute posterior mean *)
      let y_i = tensor_slice tensor mode i in
      let masked_y = Tensor.mul y_i mask in
      let m = Tensor.(
        mm (mm masked_y kr_expect) v
      ) in
      Tensor.copy_ (Tensor.select new_mean ~dim:0 ~index:i) m
    done;
    
    (new_mean, new_cov)

  (* Update sparse component posterior *)
  let update_sparse_posterior state tensor mask =
    let expect_tau = state.tau_shape /. state.tau_rate in
    
    (* Compute reconstruction *)
    let reconstruction = cp_reconstruction state.factor_means in
    let residuals = Tensor.sub tensor reconstruction in
    
    (* Update precision *)
    let new_prec = Tensor.(
      add 
        (div (full_like state.gamma_shape state.gamma_shape) state.gamma_rate)
        (full_like tensor expect_tau)
    ) |> stabilize_precision in
    
    (* Update mean *)
    let new_mean = Tensor.(
      div 
        (mul (mul residuals expect_tau) mask)
        new_prec
    ) in
    
    (new_mean, new_prec)

  (* Update lambda hyperparameters *)
  let update_lambda_posterior state =
    let n_factors = Array.length state.factor_means in
    let rank = (Tensor.shape state.factor_means.(0)).(1) in
    
    (* Compute factor matrix statistics *)
    let factor_stats = Array.map2 
      (fun mean cov ->
        let mean_term = Tensor.(
          sum (pow mean (Scalar.float 2.0))
        ) in
        let cov_term = Tensor.(
          sum (diagonal cov)
        ) in
        Tensor.add mean_term cov_term
      ) 
      state.factor_means 
      state.factor_covs in
    
    (* Compute total dimensions *)
    let total_dims = Array.fold_left (fun acc mean ->
      acc + (Tensor.shape mean).(0)
    ) 0 state.factor_means in
    
    (* Update shape and rate *)
    let new_shape = state.lambda_shape +. 
      (Float.of_int total_dims) /. 2.0 in
    
    let sum_stats = Array.fold_left Tensor.add 
      (Tensor.zeros [rank]) factor_stats in
    let new_rate = Tensor.add state.lambda_rate 
      (Tensor.div sum_stats (Tensor.full [rank] 2.0)) in
    
    (new_shape, new_rate)

  (* Update gamma hyperparameters *)
  let update_gamma_posterior state =
    let new_shape = state.gamma_shape +. 0.5 in
    let new_rate = Tensor.(
      add state.gamma_rate
        (div
          (add
            (pow state.sparse_mean (Scalar.float 2.0))
            (div (ones_like state.sparse_precision) state.sparse_precision))
          (full_like state.sparse_mean 2.0))
    ) in
    
    (new_shape, new_rate)

  (* Update noise precision *)
  let update_tau_posterior state tensor mask =
    let reconstruction = cp_reconstruction state.factor_means in
    let residuals = Tensor.(
      sub (sub tensor reconstruction) state.sparse_mean
      |> mul mask
      |> pow (Scalar.float 2.0)
      |> sum
      |> get_float1
    ) in
    
    let n_obs = Tensor.sum mask |> Tensor.get_float1 in
    let new_shape = state.tau_shape +. n_obs /. 2.0 in
    let new_rate = state.tau_rate +. residuals /. 2.0 in
    
    (new_shape, new_rate)
end

(* Model evidence computation *)
module ModelEvidence = struct
  let compute_expected_log_likelihood state tensor mask =
    let expect_tau = state.tau_shape /. state.tau_rate in
    
    (* Compute reconstruction *)
    let reconstruction = cp_reconstruction state.factor_means in
    let residuals = Tensor.(
      sub (sub tensor reconstruction) state.sparse_mean
      |> mul mask
      |> pow (Scalar.float 2.0)
    ) in
    
    let n_obs = Tensor.sum mask |> Tensor.get_float1 in
    
    (* E[log p(Y|...)] *)
    let ll = -0.5 *. expect_tau *. (Tensor.sum residuals |> Tensor.get_float1) +.
             n_obs *. 0.5 *. (Float.log expect_tau -. Float.log (2.0 *. Float.pi)) in
    ll

  let compute_kl_factors state =
    let kl_sum = ref 0.0 in
    Array.iteri (fun i mean ->
      let cov = state.factor_covs.(i) in
      let prior_mean = Tensor.zeros_like mean in
      let expect_lambda = Tensor.div 
        (Tensor.full [state.rank] state.lambda_shape) 
        state.lambda_rate in
      let prior_prec = Tensor.diag expect_lambda in
      
      let q = MultivariateNormal.create mean cov in
      let p = MultivariateNormal.create_with_precision prior_mean prior_prec in
      kl_sum := !kl_sum +. MultivariateNormal.kl_divergence q p
    ) state.factor_means;
    !kl_sum

  let compute_kl_lambda state =
    let q = Gamma.{
      shape = state.lambda_shape;
      rate = Tensor.get_float1 state.lambda_rate;
    } in
    let p = Gamma.{
      shape = 1e-6;
      rate = 1e-6;
    } in
    Gamma.kl_divergence q p

  let compute_kl_gamma state =
    let n_elements = Tensor.numel state.gamma_rate in
    let kl_sum = ref 0.0 in
    for i = 0 to n_elements - 1 do
      let q = Gamma.{
        shape = state.gamma_shape;
        rate = Tensor.get_float1 (Tensor.reshape state.gamma_rate [|-1|]) i;
      } in
      let p = Gamma.{
        shape = 1e-6;
        rate = 1e-6;
      } in
      kl_sum := !kl_sum +. Gamma.kl_divergence q p
    done;
    !kl_sum

  let compute_kl_tau state =
    let q = Gamma.{
      shape = state.tau_shape;
      rate = state.tau_rate;
    } in
    let p = Gamma.{
      shape = 1e-6;
      rate = 1e-6;
    } in
    Gamma.kl_divergence q p

  let compute_elbo state tensor mask =
    let expect_ll = compute_expected_log_likelihood state tensor mask in
    let kl_factors = compute_kl_factors state in
    let kl_lambda = compute_kl_lambda state in
    let kl_gamma = compute_kl_gamma state in
    let kl_tau = compute_kl_tau state in
    
    expect_ll -. kl_factors -. kl_lambda -. kl_gamma -. kl_tau
end

(* Inference module *)
module Inference = struct
  let update_state state tensor mask =
    (* Update factor matrices *)
    let n_modes = Array.length state.factor_means in
    let new_factors = Array.init n_modes (fun mode ->
      PosteriorUpdates.update_factor_posterior state tensor mask mode
    ) in
    
    let state = { state with
      factor_means = Array.map fst new_factors;
      factor_covs = Array.map snd new_factors;
    } in
    
    (* Update sparse component *)
    let (new_sparse_mean, new_sparse_prec) = 
      PosteriorUpdates.update_sparse_posterior state tensor mask in
    let state = { state with
      sparse_mean = new_sparse_mean;
      sparse_precision = new_sparse_prec;
    } in
    
    (* Update hyperparameters *)
    let (new_lambda_shape, new_lambda_rate) = 
      PosteriorUpdates.update_lambda_posterior state in
    let (new_gamma_shape, new_gamma_rate) = 
      PosteriorUpdates.update_gamma_posterior state in
    let (new_tau_shape, new_tau_rate) = 
      PosteriorUpdates.update_tau_posterior state tensor mask in
    
    { state with
      lambda_shape = new_lambda_shape;
      lambda_rate = new_lambda_rate;
      gamma_shape = new_gamma_shape;
      gamma_rate = new_gamma_rate;
      tau_shape = new_tau_shape;
      tau_rate = new_tau_rate;
    }

  let check_convergence state prev_elbo config =
    let elbo_diff = Float.abs (state.elbo -. prev_elbo) in
    let rel_change = elbo_diff /. Float.abs prev_elbo in
    rel_change < config.tolerance || state.iteration >= config.max_iter

  let fit tensor mask config =
    let state = ref (ModelState.create config) in
    let converged = ref false in
    
    while not !converged do
      let prev_elbo = !state.elbo in
      
      (* Update state *)
      state := update_state !state tensor mask;
      
      (* Compute ELBO *)
      let new_elbo = ModelEvidence.compute_elbo !state tensor mask in
      state := { !state with
        elbo = new_elbo;
        iteration = !state.iteration + 1;
      };
      
      (* Check convergence *)
      converged := check_convergence !state prev_elbo config
    done;
    
    !state
end

(* Predictive distribution *)
module Prediction = struct
  type prediction = {
    mean: Tensor.t;
    variance: Tensor.t;
    samples: Tensor.t array option;
  }

  let compute_mean_prediction state =
    let reconstruction = cp_reconstruction state.factor_means in
    match state.sparse_mean with
    | Some sparse -> Tensor.add reconstruction sparse
    | None -> reconstruction

  let compute_variance state =
    let tensor_shape = Tensor.shape state.factor_means.(0) in
    let n_modes = Array.length state.factor_means in
    
    (* Factor uncertainty propagation *)
    let factor_uncertainties = Array.mapi (fun mode mean ->
      let cov = state.factor_covs.(mode) in
      let other_modes = Array.init (n_modes - 1) (fun i ->
        if i < mode then state.factor_means.(i)
        else state.factor_means.(i + 1)
      ) in
      let kr_expect = khatri_rao_expectations 
        other_modes
        (Array.init (n_modes - 1) (fun i ->
          if i < mode then state.factor_covs.(i)
          else state.factor_covs.(i + 1)
        )) in
      Tensor.(mm (mm cov kr_expect) (transpose cov ~dim0:0 ~dim1:1))
    ) state.factor_means in
    
    (* Combine uncertainties *)
    let total_variance = Array.fold_left Tensor.add 
      (Tensor.zeros tensor_shape) factor_uncertainties in
    
    (* Add sparse component uncertainty if present *)
    match state.sparse_precision with
    | Some prec -> Tensor.add total_variance (safe_div 
        (Tensor.ones_like prec) prec)
    | None -> total_variance

  let sample_predictive state n_samples =
    if n_samples <= 0 then None else
    Some (Array.init n_samples (fun _ ->
      (* Sample factor matrices *)
      let sampled_factors = Array.mapi (fun i mean ->
        let cov = state.factor_covs.(i) in
        let eps = Tensor.randn (Tensor.shape mean) in
        let chol = Tensor.cholesky cov in
        Tensor.add mean (Tensor.mm chol eps)
      ) state.factor_means in
      
      (* Generate reconstruction *)
      let reconstruction = cp_reconstruction sampled_factors in
      
      (* Add sparse component if present *)
      match state.sparse_mean, state.sparse_precision with
      | Some mean, Some prec ->
          let eps = Tensor.randn (Tensor.shape mean) in
          let sparse_sample = Tensor.add mean 
            (Tensor.div eps (safe_sqrt prec)) in
          Tensor.add reconstruction sparse_sample
      | _ -> reconstruction
    ))

  let predict state ?n_samples () =
    let mean = compute_mean_prediction state in
    let variance = compute_variance state in
    let samples = match n_samples with
      | None -> None
      | Some n -> sample_predictive state n in
    { mean; variance; samples }
end