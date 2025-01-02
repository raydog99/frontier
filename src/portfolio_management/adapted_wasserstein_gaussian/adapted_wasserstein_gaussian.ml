open Torch

module Gaussian = struct
  type t = {
    mean: Tensor.t;
    cov: Tensor.t;
  }

  let create mean cov = {mean; cov}

  let cholesky cov =
    Torch.linalg_cholesky cov

  let sample g n =
    let d = Tensor.shape g.mean |> List.hd in
    let eps = Torch.randn [n; d] in
    let L = cholesky g.cov in
    Torch.matmul eps L |> Torch.add_ g.mean

  let log_density g x =
    let d = Tensor.shape g.mean |> List.hd |> float_of_int in
    let x_centered = Torch.sub x g.mean in
    let L = cholesky g.cov in
    let solved = Torch.triangular_solve x_centered L ~upper:false ~transpose:false ~unitriangular:false in
    let log_det = Torch.diagonal L |> Torch.log |> Torch.sum in
    let quad_form = Torch.square solved |> Torch.sum in
    Torch.sub (Torch.mul_scalar (-0.5) quad_form) 
      (Torch.add_scalar (Torch.mul_scalar d (Float.log (2. *. Float.pi))) log_det)
end

let safe_cholesky mat =
  let eps = 1e-6 in
  let jitter = eye (List.hd (Tensor.shape mat)) |> mul_scalar eps in
  try
    linalg_cholesky mat
  with _ ->
    linalg_cholesky (add mat jitter)

let matrix_sqrt mat =
  let u, s, v = svd mat in
  let s_sqrt = sqrt s in
  mm (mm u (diag s_sqrt)) (transpose v ~dim0:0 ~dim1:1)

let matrix_invsqrt mat =
  let u, s, v = svd mat in
  let s_inv_sqrt = div (ones_like s) (sqrt (add s (ones_like s |> mul_scalar 1e-10))) in
  mm (mm u (diag s_inv_sqrt)) (transpose v ~dim0:0 ~dim1:1)

let extract_block mat start_row start_col size =
  narrow (narrow mat ~dim:0 ~start:start_row ~length:size)
         ~dim:1 ~start:start_col ~length:size

let set_block target source start_row start_col =
  let size = List.hd (Tensor.shape source) in
  let block = narrow (narrow target ~dim:0 ~start:start_row ~length:size)
                    ~dim:1 ~start:start_col ~length:size in
  copy_ block source

let is_positive_definite mat =
  let e = linalg_eigh mat |> fst in
  all (gt e (zeros_like e))

let kron a b =
  let m1, n1 = match Tensor.shape a with
    | [m; n] -> m, n
    | _ -> failwith "Invalid shape for first matrix" in
  let m2, n2 = match Tensor.shape b with
    | [m; n] -> m, n
    | _ -> failwith "Invalid shape for second matrix" in
  
  let result = zeros [m1 * m2; n1 * n2] in
  
  for i = 0 to m1 - 1 do
    for j = 0 to n1 - 1 do
      let block = mul_scalar (get a [|i; j|]) b in
      set_block result block (i * m2) (j * n2)
    done
  done;
  result

let matrix_log mat =
  let eps = 1e-10 in
  let e, v = linalg_eigh mat in
  let e = max e (full_like e eps) in
  let log_e = log e in
  mm (mm v (diag log_e)) (transpose v ~dim0:0 ~dim1:1)

let matrix_exp mat =
  let e, v = linalg_eigh mat in
  let exp_e = exp e in
  mm (mm v (diag exp_e)) (transpose v ~dim0:0 ~dim1:1)

let matrix_power_pd mat power =
  let e, v = linalg_eigh mat in
  let eps = 1e-10 in
  let e = max e (full_like e eps) in
  let power_e = pow e (full_like e power) in
  mm (mm v (diag power_e)) (transpose v ~dim0:0 ~dim1:1)

let solve_sylvester a b c =
  let m = List.hd (shape a) in
  let n = List.hd (shape b) in
  
  let eye_n = eye n in
  let eye_m = eye m in
  let kron_a = kron eye_n a in
  let kron_b = kron (transpose b ~dim0:0 ~dim1:1) eye_m in
  let coeff = add kron_a kron_b in
  
  let vec_c = reshape c [m * n] in
  let vec_x = solve coeff vec_c in
  
  reshape vec_x [m; n]

let matrix_gmean a b =
  let sqrt_a = matrix_sqrt a in
  let sqrt_a_inv = inverse sqrt_a in
  let inner = mm (mm sqrt_a_inv b) sqrt_a_inv in
  let inner_sqrt = matrix_sqrt inner in
  mm (mm sqrt_a inner_sqrt) sqrt_a_inv

let project_psd mat =
  let e, v = linalg_eigh mat in
  let e = max e (zeros_like e) in
  mm (mm v (diag e)) (transpose v ~dim0:0 ~dim1:1)

let frechet_derivative f mat h =
  let eps = 1e-8 in
  let forward = f (add mat (mul_scalar eps h)) in
  let backward = f (sub mat (mul_scalar eps h)) in
  div (sub forward backward) (full_like forward (2.0 *. eps))

let matrix_gradient f mat =
  let n = List.hd (shape mat) in
  let grad = zeros [n; n] in
  
  for i = 0 to n-1 do
    for j = 0 to n-1 do
      let h = zeros [n; n] in
      set h i j (ones []) 1.0;
      let deriv = frechet_derivative f mat h in
      set grad i j deriv 1.0
    done
  done;
  grad

let matrix_hessian f mat =
  let n = List.hd (shape mat) in
  let hess = zeros [n; n; n; n] in
  
  for i = 0 to n-1 do
    for j = 0 to n-1 do
      let h1 = zeros [n; n] in
      set h1 i j (ones []) 1.0;
      let grad_h1 = frechet_derivative f mat h1 in
      
      for k = 0 to n-1 do
        for l = 0 to n-1 do
          let h2 = zeros [n; n] in
          set h2 k l (ones []) 1.0;
          let deriv = frechet_derivative 
            (fun m -> frechet_derivative f m h1) mat h2 in
          set hess i j k l deriv 1.0
        done
      done
    done
  done;
  hess

let kl_divergence pi mu_nu =
  let log_pi = Gaussian.log_density pi in
  let log_prod = Torch.add 
    (Gaussian.log_density mu_nu.mean) 
    (Gaussian.log_density mu_nu.cov) in
  Torch.mean (Torch.sub log_pi log_prod)

let quadratic_cost x y =
  let diff = Torch.sub x y in
  Torch.sum (Torch.square diff) ~dim:[1]

let entropic_wasserstein_2 ?(lambda=1.0) mu nu =
  let n_samples = 1000 in
  let samples_mu = Gaussian.sample mu n_samples in
  let samples_nu = Gaussian.sample nu n_samples in
  
  let cost = quadratic_cost samples_mu samples_nu in
  let pi = Gaussian.create samples_mu samples_nu in
  let entropy = kl_divergence pi (Gaussian.create mu.mean nu.cov) in
  
  Torch.add cost (Torch.mul_scalar lambda entropy)

let f_lambda lambda x =
  if Tensor.equal x (zeros_like x) then zeros_like x
  else
    let inner = sqrt (add (mul_scalar 16. (square x)) 
                        (mul_scalar (lambda *. lambda) (ones_like x))) in
    div inner (mul_scalar 4. x)

let compute_trace_terms a b d s =
  let tr_ab = add (trace a) (trace b) in
  let tr_ds = mul_scalar (-2.) (trace (mm d s)) in
  add tr_ab tr_ds

let compute_det_term lambda d =
  if lambda > 0. then
    let i = eye (List.hd (shape d)) in
    let d2 = square d in
    mul_scalar (-0.5 *. lambda) (logdet (sub i d2))
  else
    zeros []

let adapted_wasserstein_2 ?(lambda=0.) mu nu =
  let a = mu.Gaussian.cov in
  let b = nu.Gaussian.cov in
  
  (* Get cholesky factors *)
  let l = safe_cholesky a in
  let m = safe_cholesky b in
  
  (* Get dimensions *)
  let d = List.hd (shape mu.Gaussian.mean) in
  let t = (List.hd (shape a)) / d in
  
  (* Compute M'L *)
  let ml = mm (transpose m ~dim0:0 ~dim1:1) l in
  let u, s, v = svd ml in
  
  (* Apply f_lambda *)
  let d = f_lambda lambda s in
  
  (* Compute mean term *)
  let mean_diff = sub mu.Gaussian.mean nu.Gaussian.mean in
  let mean_term = sum (square mean_diff) in
  
  (* Compute trace terms *)
  let trace_terms = compute_trace_terms a b d s in
  
  (* Compute determinant term *)
  let det_term = compute_det_term lambda d in
  
  add mean_term (add trace_terms det_term)

let construct_optimal_coupling mu nu lambda =
  let l = safe_cholesky mu.Gaussian.cov in
  let m = safe_cholesky nu.Gaussian.cov in
  
  let ml = mm (transpose m ~dim0:0 ~dim1:1) l in
  let u, s, v = svd ml in
  
  let d = f_lambda lambda s in
  let p = mm (mm u (diag d)) (transpose v ~dim0:0 ~dim1:1) in
  
  let cov = cat [
    cat [mm l (transpose l ~dim0:0 ~dim1:1); 
         mm (mm l p) (transpose m ~dim0:0 ~dim1:1)] ~dim:1;
    cat [mm (mm m (transpose p ~dim0:0 ~dim1:1)) (transpose l ~dim0:0 ~dim1:1);
         mm m (transpose m ~dim0:0 ~dim1:1)] ~dim:1
  ] ~dim:0 in
  
  let mean = cat [mu.Gaussian.mean; nu.Gaussian.mean] ~dim:0 in
  Gaussian.create mean cov

module DPControl = struct
  type value_function = {
    value: Tensor.t;
    gradient: Tensor.t;
  }

  type policy = {
    coupling: Gaussian.t;
    next_state: Tensor.t -> Tensor.t * Tensor.t;
  }

  module StageCost = struct
    let quadratic x y =
      sub x y |> square |> sum ~dim:[0]

    let entropic pi mu_nu lambda =
      let kl = CouplingDefinitions.kl_divergence pi mu_nu in
      mul_scalar lambda kl

    let total x y pi mu_nu lambda =
      let quad = quadratic x y in
      let ent = entropic pi mu_nu lambda in
      add quad ent
  end

  module ConditionalLaw = struct
    let compute_gaussian g x t =
      let l = safe_cholesky g.Gaussian.cov in
      let d = List.hd (Tensor.shape g.Gaussian.mean) in
      
      (* Extract blocks *)
      let l11 = extract_block l 0 0 (t * d) in
      let l21 = extract_block l (t * d) 0 d in
      let l22 = extract_block l (t * d) (t * d) d in
      
      (* Compute conditional parameters *)
      let l11_inv = inverse l11 in
      let mean_diff = sub x (narrow g.Gaussian.mean ~dim:0 ~start:0 ~length:(t * d)) in
      
      let cond_mean = add 
        (narrow g.Gaussian.mean ~dim:0 ~start:(t * d) ~length:d)
        (mm (mm l21 l11_inv) mean_diff) in
        
      let cond_cov = sub 
        (mm l22 (transpose l22 ~dim0:0 ~dim1:1))
        (mm (mm l21 l11_inv) (transpose l21 ~dim0:0 ~dim1:1)) in
        
      Gaussian.create cond_mean cond_cov

    let compute_coupling pi x y t =
      let joint = cat [x; y] ~dim:0 in
      compute_gaussian pi joint t
  end

  module ValueIteration = struct
    let backward mu nu lambda =
      let d = List.hd (shape mu.Gaussian.mean) in
      let t = (List.hd (shape mu.Gaussian.cov)) / d in
      
      let rec backward_step time values policies =
        if time < 0 then List.rev values, List.rev policies
        else
          (* Get conditional distributions *)
          let mu_t = ConditionalLaw.compute_gaussian mu (zeros [time * d]) time in
          let nu_t = ConditionalLaw.compute_gaussian nu (zeros [time * d]) time in
          
          (* Optimize stage coupling *)
          let coupling = construct_optimal_coupling 
                         mu_t nu_t lambda in
          
          (* Compute stage cost *)
          let v_next = match values with
            | [] -> zeros []
            | v::_ -> v.value in
            
          let cost = StageCost.total mu_t.mean nu_t.mean coupling 
                      (Gaussian.create mu_t.mean nu_t.mean) lambda in
          let value = add cost v_next in
          
          (* Create policy *)
          let policy = {
            coupling;
            next_state = (fun state -> 
              let next_x = add (mm state (transpose mu_t.mean ~dim0:0 ~dim1:1)) mu_t.mean in
              let next_y = add (mm state (transpose nu_t.mean ~dim0:0 ~dim1:1)) nu_t.mean in
              next_x, next_y)
          } in
          
          let value_func = {
            value;
            gradient = matrix_gradient 
                        (fun _ -> value) mu_t.mean
          } in
          
          backward_step (time-1) (value_func :: values) (policy :: policies)
      in
      
      backward_step (t-1) [] []
  end
end