open Torch

module MatrixOps = struct
  let svd x =
    let u, s, v = Tensor.svd x ~some:false in
    (u, s, v)
    
  let matrix_multiply a b =
    Tensor.mm a b
    
  let transpose x =
    Tensor.transpose x ~dim0:0 ~dim1:1
    
  let l2_norm x =
    Tensor.norm x ~p:(Scalar.F 2.0) ~dim:[0] ~keepdim:true
    
  let make_diagonal x =
    Tensor.diag x
end

module StiefelManifold = struct
  let project_tangent v =
    let vt = MatrixOps.transpose v in
    let proj = Tensor.mm v vt in
    let eye = Tensor.eye (Tensor.size v 1) in
    Tensor.sub eye proj
    
  let manifold_gradient gradient v =
    let proj = project_tangent v in
    Tensor.mm gradient proj
    
  let retract x =
    let u, s, v = MatrixOps.svd x in
    Tensor.mm u (MatrixOps.transpose v)
    
  let compute_constraint_violation model k =
    let u_k = Tensor.select model.u ~dim:1 ~index:k in
    let v_k = Tensor.select model.v ~dim:1 ~index:k in
    
    let ut_u = Tensor.mm (MatrixOps.transpose u_k) u_k in
    let vt_v = Tensor.mm (MatrixOps.transpose v_k) v_k in
    let eye = Tensor.eye 1 in
    
    let u_violation = Tensor.norm 
      (Tensor.sub ut_u eye) ~p:(Scalar.F 2.0) in
    let v_violation = Tensor.norm 
      (Tensor.sub vt_v eye) ~p:(Scalar.F 2.0) in
      
    Tensor.to_float0_exn (Tensor.add u_violation v_violation)
end

module Sofar = struct
  type model = {
    u: Tensor.t;
    d: Tensor.t;
    v: Tensor.t;
    rank: int;
    x: Tensor.t;
    y: Tensor.t;
    sigma: Tensor.t;
    theta: Tensor.t;
  }

  let init_model n_features n_responses rank =
    let u = Tensor.randn [n_features; rank] in
    let d = Tensor.randn [rank] in
    let v = Tensor.randn [n_responses; rank] in
    let x = Tensor.zeros [0; n_features] in
    let y = Tensor.zeros [0; n_responses] in
    let sigma = Tensor.zeros [n_features; n_features] in
    let theta = Tensor.zeros [n_features; n_features] in
    { u; d; v; rank; x; y; sigma; theta }

  let estimate x y rank lambda_d lambda_u lambda_v =
    let n_features = Tensor.size x 1 in
    let n_responses = Tensor.size y 1 in
    
    let model = init_model n_features n_responses rank in
    
    (* Initialize optimization variables *)
    let u = Tensor.requires_grad_ model.u in
    let d = Tensor.requires_grad_ model.d in
    let v = Tensor.requires_grad_ model.v in
    
    let loss_fn () =
      let pred = Tensor.mm (Tensor.mm x u) 
        (Tensor.mm (MatrixOps.make_diagonal d) 
         (MatrixOps.transpose v)) in
      let mse = Tensor.mse_loss pred y 
        ~reduction:Reduction.Mean in
      
      (* Add regularization *)
      let reg_d = Tensor.mul_scalar 
        (Tensor.norm d ~p:(Scalar.F 1.0)) lambda_d in
      let reg_u = Tensor.mul_scalar 
        (Tensor.norm u ~p:(Scalar.F 1.0)) lambda_u in
      let reg_v = Tensor.mul_scalar 
        (Tensor.norm v ~p:(Scalar.F 1.0)) lambda_v in
      
      Tensor.add (Tensor.add (Tensor.add mse reg_d) reg_u) reg_v
    in
    
    let optimizer = Optimizer.adam [u; d; v] ~lr:1e-3 in
    
    (* Training loop *)
    for _ = 1 to 1000 do
      Optimizer.zero_grad optimizer;
      let loss = loss_fn () in
      Tensor.backward loss;
      Optimizer.step optimizer
    done;
    
    (* Compute sigma and theta *)
    let n = float_of_int (Tensor.size x 0) in
    let sigma = Tensor.div_scalar 
      (Tensor.mm (MatrixOps.transpose x) x) n in
    let theta = NumericalStability.stable_inverse
      sigma NumericalStability.default_params in
    
    { u = Tensor.detach u; 
      d = Tensor.detach d; 
      v = Tensor.detach v; 
      rank;
      x;
      y;
      sigma;
      theta }

  let estimate_with_constraints x y rank lambda_d lambda_u lambda_v constraint_lambda =
    let model = estimate x y rank lambda_d lambda_u lambda_v in
    let n_features = Tensor.size x 1 in
    let n_responses = Tensor.size y 1 in
    
    (* Initialize new optimization variables *)
    let u = Tensor.requires_grad_ model.u in
    let d = Tensor.requires_grad_ model.d in
    let v = Tensor.requires_grad_ model.v in
    
    let loss_fn () =
      let pred = Tensor.mm (Tensor.mm x u) 
        (Tensor.mm (MatrixOps.make_diagonal d) 
         (MatrixOps.transpose v)) in
      let mse = Tensor.mse_loss pred y 
        ~reduction:Reduction.Mean in
      
      (* Regular regularization *)
      let reg_d = Tensor.mul_scalar 
        (Tensor.norm d ~p:(Scalar.F 1.0)) lambda_d in
      let reg_u = Tensor.mul_scalar 
        (Tensor.norm u ~p:(Scalar.F 1.0)) lambda_u in
      let reg_v = Tensor.mul_scalar 
        (Tensor.norm v ~p:(Scalar.F 1.0)) lambda_v in
        
      (* Orthogonality constraints *)
      let ut_u = Tensor.mm (MatrixOps.transpose u) u in
      let vt_v = Tensor.mm (MatrixOps.transpose v) v in
      let eye_p = Tensor.eye rank in
      let eye_q = Tensor.eye rank in
      
      let ortho_u = Tensor.norm 
        (Tensor.sub ut_u eye_p) ~p:(Scalar.F 2.0) in
      let ortho_v = Tensor.norm 
        (Tensor.sub vt_v eye_q) ~p:(Scalar.F 2.0) in
      
      let constraints = Tensor.mul_scalar 
        (Tensor.add ortho_u ortho_v) constraint_lambda in
      
      Tensor.add 
        (Tensor.add (Tensor.add (Tensor.add mse reg_d) reg_u) reg_v)
        constraints
    in
    
    let optimizer = Optimizer.adam [u; d; v] ~lr:1e-3 in
    
    (* Training loop with manifold projection *)
    for _ = 1 to 1000 do
      Optimizer.zero_grad optimizer;
      let loss = loss_fn () in
      Tensor.backward loss;
      
      (* Project gradients onto manifold *)
      let u_grad = Tensor.autograd loss u in
      let v_grad = Tensor.autograd loss v in
      
      let proj_u = StiefelManifold.manifold_gradient u_grad u in
      let proj_v = StiefelManifold.manifold_gradient v_grad v in
      
      (* Update with retraction *)
      let new_u = StiefelManifold.retract 
        (Tensor.sub u proj_u) in
      let new_v = StiefelManifold.retract 
        (Tensor.sub v proj_v) in
        
      Tensor.copy_ u new_u;
      Tensor.copy_ v new_v;
      
      Optimizer.step optimizer
    done;
    
    { model with 
      u = Tensor.detach u;
      d = Tensor.detach d;
      v = Tensor.detach v }
end

module NeymanScore = struct
  type score = {
    e_k: Tensor.t;
    m_vk: Tensor.t;
    z_kk: Tensor.t;
  }

  let modified_score_function model k x y =
    let n = Tensor.size x 0 in
    let u_k = Tensor.select model.u ~dim:1 ~index:k in
    let v_k = Tensor.select model.v ~dim:1 ~index:k in
    
    (* Compute z_kk *)
    let z_kk = Tensor.mm (MatrixOps.transpose u_k) 
      (Tensor.mm model.sigma u_k) in
    
    (* Construct C_k *)
    let c_k = Tensor.zeros [Tensor.size model.u 0; Tensor.size model.v 0] in
    for i = 0 to model.rank - 1 do
      if i <> k then begin
        let u_i = Tensor.select model.u ~dim:1 ~index:i in
        let v_i = Tensor.select model.v ~dim:1 ~index:i in
        let prod = Tensor.mm u_i (MatrixOps.transpose v_i) in
        c_k += prod
      end
    done;
    
    (* Compute M_vk *)
    let m_vk = Tensor.mm (Tensor.div_scalar model.sigma 
      (Tensor.to_float0_exn z_kk)) c_k in
    
    (* Compute score components *)
    let resid = Tensor.sub y (Tensor.mm x 
      (Tensor.mm model.u (Tensor.mm (MatrixOps.make_diagonal model.d) 
       (MatrixOps.transpose model.v)))) in
    let score_u = Tensor.mm (MatrixOps.transpose x) 
      (Tensor.mm resid v_k) in
    let score_noise = Tensor.mm m_vk 
      (Tensor.mm (MatrixOps.transpose resid) (Tensor.mm x u_k)) in
    
    { e_k = Tensor.sub score_u score_noise;
      m_vk;
      z_kk }
    
  let approximation_error score eta_k eta_star_k =
    let diff = Tensor.sub score.e_k 
      (modified_score_function score.m_vk score.z_kk eta_star_k) in
    Tensor.norm diff ~p:(Scalar.F 1.0)
end

module NumericalStability = struct
  type stability_params = {
    eps: float;
    max_cond_number: float;
    min_eigenval: float;
  }

  let default_params = {
    eps = 1e-10;
    max_cond_number = 1e6;
    min_eigenval = 1e-8;
  }

  let check_conditioning matrix params =
    let eigenvals = Tensor.linalg_eigvalsh matrix in
    let max_eval = Tensor.max eigenvals |> Tensor.to_float0_exn in
    let min_eval = Tensor.min eigenvals |> Tensor.to_float0_exn in
    let cond_number = max_eval /. min_eval in
    
    cond_number < params.max_cond_number &&
    min_eval > params.min_eigenval
    
  let stable_inverse matrix params =
    let n = Tensor.size matrix 0 in
    let eigenvals = Tensor.linalg_eigvalsh matrix in
    let min_eval = Tensor.min eigenvals |> Tensor.to_float0_exn in
    
    if min_eval < params.min_eigenval then
      let ridge = Tensor.mul_scalar (Tensor.eye n) params.eps in
      Tensor.inverse (Tensor.add matrix ridge)
    else
      Tensor.inverse matrix
end

module RobustConditionChecker = struct
  type condition_result = {
    satisfied: bool;
    error_margin: float;
    confidence: float;
    message: string;
  }

  let check_sparse_eigenvalues sigma s rho_l rho_u params =
    let n = Tensor.size sigma 0 in
    let valid = ref true in
    let min_margin = ref Float.infinity in
    
    (* Generate s-sparse test vectors *)
    let test_vectors = ref [] in
    for _ = 1 to 100 do
      let v = Tensor.zeros [n] in
      let indices = Array.init n (fun i -> i)
        |> Array.to_seq
        |> Seq.take s
        |> List.of_seq in
      List.iter (fun i ->
        Tensor.set_ v ~index:i (Tensor.full [1] 1.0)
      ) indices;
      test_vectors := v :: !test_vectors
    done;
    
    (* Check condition for each test vector *)
    List.iter (fun v ->
      let sigma_v = Tensor.mm sigma v in
      let ratio = Tensor.div (MatrixOps.l2_norm sigma_v)
        (MatrixOps.l2_norm v) in
      let ratio_val = Tensor.to_float0_exn ratio in
      
      valid := !valid && ratio_val >= rho_l && ratio_val <= rho_u;
      min_margin := min !min_margin
        (min (ratio_val -. rho_l) (rho_u -. ratio_val))
    ) !test_vectors;
    
    (* Compute confidence based on sampling *)
    let confidence = 1.0 -. 
      (1.0 /. float_of_int (List.length !test_vectors)) in
    
    { satisfied = !valid;
      error_margin = !min_margin;
      confidence;
      message = "Sparse eigenvalue condition check" }

  let check_singular_separation d gamma_1 r_star delta_n params =
    let valid = ref true in
    let min_separation = ref Float.infinity in
    let eigenval_ratios = ref [] in
    
    for i = 0 to r_star - 2 do
      let d_i = Tensor.get d ~index:i |> Tensor.to_float0_exn in
      let d_i1 = Tensor.get d ~index:(i+1) |> Tensor.to_float0_exn in
      let separation = (d_i *. d_i -. d_i1 *. d_i1) /. 
        (d_i *. d_i) in
      
      valid := !valid && separation >= gamma_1;
      min_separation := min !min_separation separation;
      eigenval_ratios := (d_i1 /. d_i) :: !eigenval_ratios
    done;
    
    let d_r = Tensor.get d ~index:(r_star-1) |> Tensor.to_float0_exn in
    let rank_reliable = float_of_int r_star *. delta_n < d_r in
    
    let confidence = if !min_separation > gamma_1 then
      (1.0 -. exp(-. (!min_separation -. gamma_1) /. gamma_1))
    else
      0.0 in
    
    { satisfied = !valid && rank_reliable;
      error_margin = !min_separation -. gamma_1;
      confidence;
      message = "Singular value separation check" }

  let check_orthogonality sigma u k n params =
    let correlations = ref [] in
    let max_correlation = ref 0.0 in
    let u_k = Tensor.select u ~dim:1 ~index:k in
    
    for j = 0 to Tensor.size u 1 - 1 do
      if j <> k then begin
        let u_j = Tensor.select u ~dim:1 ~index:j in
        let corr = Tensor.mm (MatrixOps.transpose u_j)
          (Tensor.mm sigma u_k) in
        let corr_val = Tensor.to_float0_exn (Tensor.abs corr) in
        
        correlations := corr_val :: !correlations;
        max_correlation := max !max_correlation corr_val
      end
    done;
    
    let threshold = 1.0 /. sqrt (float_of_int n) in
    let margin = threshold -. !max_correlation in
    
    let confidence = if margin > 0.0 then
      (1.0 -. exp(-. margin *. float_of_int n))
    else
      0.0 in
    
    { satisfied = margin > 0.0;
      error_margin = margin;
      confidence;
      message = "Orthogonality condition check" }
end

module ErrorTracking = struct
  type error_bounds = {
    estimation_error: float;
    numerical_error: float;
    total_error: float;
    confidence_level: float;
  }

  let track_error results params =
    let max_estimation_error = List.fold_left
      (fun acc result -> max acc (1.0 -. result.confidence))
      0.0 results in
    
    let numerical_error = params.eps *.
      float_of_int (List.length results) in
    
    let total_error = max_estimation_error +. numerical_error in
    let confidence = 1.0 -. total_error in
    
    { estimation_error = max_estimation_error;
      numerical_error;
      total_error;
      confidence_level = confidence }
end

module SofariStrict = struct
  type model = {
    sofar: Sofar.model;
    sigma: Tensor.t;
    theta: Tensor.t;
  }

  let construct_m_strict model k =
    let u_k = Tensor.select model.sofar.u ~dim:1 ~index:k in
    let z_kk = Tensor.mm (MatrixOps.transpose u_k) 
      (Tensor.mm model.sigma u_k) in
    
    (* Construct C_k excluding kth component *)
    let c_k = Tensor.zeros [Tensor.size model.sofar.u 0; 
                          Tensor.size model.sofar.v 0] in
    for i = 0 to model.sofar.rank - 1 do
      if i <> k then begin
        let u_i = Tensor.select model.sofar.u ~dim:1 ~index:i in
        let v_i = Tensor.select model.sofar.v ~dim:1 ~index:i in
        let prod = Tensor.mm u_i (MatrixOps.transpose v_i) in
        c_k += prod
      end
    done;
    
    let m_vk = Tensor.mm (Tensor.div_scalar model.sigma 
      (Tensor.to_float0_exn z_kk)) c_k in
    (m_vk, z_kk)
    
  let construct_w_strict model k m_vk z_kk =
    let u_k = Tensor.select model.sofar.u ~dim:1 ~index:k in
    let v_k = Tensor.select model.sofar.v ~dim:1 ~index:k in
    
    (* Construct U_k excluding kth component *)
    let u_indices = List.init model.sofar.rank 
      (fun i -> if i <> k then Some i else None) in
    let u_k_matrix = Tensor.cat 
      (List.filter_map (fun i -> 
        match i with 
        | Some idx -> Some (Tensor.select model.sofar.u ~dim:1 ~index:idx)
        | None -> None) u_indices) ~dim:1 in
        
    (* Calculate inverse components *)
    let uk_sigma = Tensor.mm model.sigma u_k_matrix in
    let uk_sigma_uk = Tensor.mm (MatrixOps.transpose u_k_matrix) uk_sigma in
    let eye = Tensor.eye (model.sofar.rank - 1) in
    let term1 = Tensor.sub eye 
      (Tensor.div_scalar uk_sigma_uk (Tensor.to_float0_exn z_kk)) in
    let term1_inv = NumericalStability.stable_inverse
      term1 NumericalStability.default_params in
    
    (* Combine terms *)
    let w_base = Tensor.eye (Tensor.size model.sofar.u 0) in
    let correction = Tensor.mm uk_sigma 
      (Tensor.mm term1_inv (MatrixOps.transpose u_k_matrix)) in
    Tensor.mm model.theta 
      (Tensor.add w_base (Tensor.div_scalar correction 
       (Tensor.to_float0_exn z_kk)))
    
  let infer_strict x y k model =
    let m_vk, z_kk = construct_m_strict model k in
    let w = construct_w_strict model k m_vk z_kk in
    let score = NeymanScore.modified_score_function model.sofar k x y in
    
    (* Compute debiased estimate *)
    let u_k = Tensor.select model.sofar.u ~dim:1 ~index:k in
    let debiased_u = Tensor.sub u_k (Tensor.mm w score.e_k) in
    
    (* Estimate variance *)
    let sigma_e = Tensor.eye (Tensor.size y 1) in
    let variance_term = Tensor.mm w 
      (Tensor.add 
        (Tensor.mm (Tensor.mul_scalar m_vk 
         (Tensor.to_float0_exn z_kk))
         (Tensor.mm sigma_e (MatrixOps.transpose m_vk)))
        (Tensor.mm (Tensor.mm model.sigma 
         (Tensor.mul sigma_e 
          (MatrixOps.transpose (Tensor.select model.sofar.v 
           ~dim:1 ~index:k)))) w)) in
        
    (debiased_u, variance_term)
end

module SofariRelaxed = struct
  let remove_previous_layers x y model k =
    let pred = ref (Tensor.zeros_like y) in
    for i = 0 to k - 1 do
      let u_i = Tensor.select model.u ~dim:1 ~index:i in
      let v_i = Tensor.select model.v ~dim:1 ~index:i in
      let d_i = Tensor.select model.d ~dim:0 ~index:i in
      pred := Tensor.add !pred 
        (Tensor.mm (Tensor.mm x u_i)
         (Tensor.mm (Tensor.reshape d_i ~shape:[1; 1])
          (MatrixOps.transpose v_i)))
    done;
    Tensor.sub y !pred
    
  let construct_residual_matrices model k =
    let r = model.rank in
    
    let c_star_2 = ref (Tensor.zeros [Tensor.size model.u 0;
                                    Tensor.size model.v 0]) in
    for i = k + 1 to r - 1 do
      let u_i = Tensor.select model.u ~dim:1 ~index:i in
      let v_i = Tensor.select model.v ~dim:1 ~index:i in
      c_star_2 := Tensor.add !c_star_2
        (Tensor.mm u_i (MatrixOps.transpose v_i))
    done;
    
    let c_star_2j = Array.make (r - k)
      (Tensor.zeros [Tensor.size model.u 0;
                    Tensor.size model.v 0]) in
    for j = k + 1 to r - 1 do
      let sum = ref (Tensor.zeros [Tensor.size model.u 0;
                                 Tensor.size model.v 0]) in
      for i = k + 1 to r - 1 do
        if i <> j then begin
          let u_i = Tensor.select model.u ~dim:1 ~index:i in
          let v_i = Tensor.select model.v ~dim:1 ~index:i in
          sum := Tensor.add !sum
            (Tensor.mm u_i (MatrixOps.transpose v_i))
        end
      done;
      c_star_2j.(j-k-1) <- !sum
    done;
    
    (!c_star_2, c_star_2j)
    
  let construct_m_relaxed model k c_star_2 =
    let u_k = Tensor.select model.u ~dim:1 ~index:k in
    let z_kk = Tensor.mm (MatrixOps.transpose u_k)
      (Tensor.mm model.sigma u_k) in
    let m_vk = Tensor.mm (Tensor.div_scalar model.sigma
      (Tensor.to_float0_exn z_kk)) c_star_2 in
    (m_vk, z_kk)
    
  let infer_relaxed x y k model =
    (* Remove previous layers *)
    let y_residual = remove_previous_layers x y model k in
    
    (* Get residual matrices *)
    let c_star_2, c_star_2j = construct_residual_matrices model k in
    
    (* Construct score function *)
    let m_vk, z_kk = construct_m_relaxed model k c_star_2 in
    let score = NeymanScore.modified_score_function model k x y_residual in
    
    (* Construct debiasing matrix *)
    let w = SofariStrict.construct_w_strict 
      { sofar = model; sigma = model.sigma; theta = model.theta } 
      k m_vk z_kk in
    
    (* Compute debiased estimates *)
    let u_k = Tensor.select model.u ~dim:1 ~index:k in
    let debiased_u = Tensor.sub u_k (Tensor.mm w score.e_k) in
    
    (* Estimate variance *)
    let sigma_e = Tensor.eye (Tensor.size y 1) in
    let variance = Tensor.mm w
      (Tensor.add
        (Tensor.mm (Tensor.mul_scalar m_vk 
         (Tensor.to_float0_exn z_kk))
         (Tensor.mm sigma_e (MatrixOps.transpose m_vk)))
        (Tensor.mm (Tensor.mm model.sigma
         (Tensor.mul sigma_e
          (MatrixOps.transpose (Tensor.select model.v
           ~dim:1 ~index:k)))) w)) in
    
    (debiased_u, variance)
    
  let infer_k1 = infer_relaxed  (* Special case k=1 uses same procedure *)
end

module AsymptoticApproximations = struct
  type approximation_quality = {
    bias: float;
    variance_ratio: float;
    convergence_rate: float;
    sample_size_requirement: int;
  }

  let verify_asymptotic_normality distribution n p =
    let samples = distribution |> Tensor.reshape ~shape:[-1] in
    
    (* Compute empirical moments *)
    let mean = Tensor.mean samples ~dim:[0] in
    let centered = Tensor.sub samples mean in
    let var = Tensor.mean (Tensor.mul centered centered) ~dim:[0] in
    let skew = Tensor.mean (Tensor.pow centered (Tensor.of_float 3.0)) ~dim:[0] in
    let kurt = Tensor.mean (Tensor.pow centered (Tensor.of_float 4.0)) ~dim:[0] in
    
    (* Check convergence rates *)
    let mean_rate = Tensor.to_float0_exn (Tensor.abs mean) *. 
      sqrt (float_of_int n) in
    let var_dev = Tensor.to_float0_exn 
      (Tensor.sub var (Tensor.of_float 1.0)) in
    let skew_rate = Tensor.to_float0_exn (Tensor.abs skew) *. 
      sqrt (float_of_int n) in
    let kurt_dev = Tensor.to_float0_exn 
      (Tensor.sub kurt (Tensor.of_float 3.0)) *. float_of_int n in
    
    (* Required sample size for different precision levels *)
    let required_n = max
      (int_of_float (10.0 *. log (float_of_int p)))
      (int_of_float (sqrt (float_of_int n) *. kurt_dev)) in
    
    { bias = Tensor.to_float0_exn mean;
      variance_ratio = Tensor.to_float0_exn var;
      convergence_rate = max mean_rate (max skew_rate kurt_dev);
      sample_size_requirement = required_n }

  let compute_enhanced_delta_n model n p q =
    let r = model.rank in
    let s_u = Tensor.count_nonzero model.u in
    let s_v = Tensor.count_nonzero model.v in
    
    (* Compute eta_n *)
    let d_ratio_sum = ref 0.0 in
    let d_1 = Tensor.get model.d ~index:0 |> Tensor.to_float0_exn in
    for j = 0 to r - 1 do
      let d_j = Tensor.get model.d ~index:j |> Tensor.to_float0_exn in
      d_ratio_sum := !d_ratio_sum +. (d_1 /. d_j) ** 2.0
    done;
    
    let eta_n = 1.0 +. sqrt !d_ratio_sum in
    let s_max = max (Tensor.to_int0_exn s_u) (Tensor.to_int0_exn s_v) in
    
    (* Enhanced computation with higher-order terms *)
    let term1 = sqrt (float_of_int (r + s_max)) in
    let term2 = eta_n *. eta_n in
    let term3 = sqrt (log (float_of_int (p * q)) /. float_of_int n) in
    
    let correction1 = log (1.0 +. log (float_of_int (p * q))) /. 
      float_of_int n in
    let correction2 = float_of_int r *. term2 *. term3 in
    
    term1 *. term2 *. term3 *. (1.0 +. correction1 +. correction2)
end