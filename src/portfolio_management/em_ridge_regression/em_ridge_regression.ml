open Torch

type model_params = {
  beta: tensor;
  gamma: float;
  delta: float;
  prior_var: tensor;
}

type latent_vars = {
  z: tensor;              
  omega: tensor;          
  z_tilde: tensor option; 
}

type model_state = {
  params: model_params;
  latent: latent_vars;
}

type chain_stats = {
  acceptance_rates: float array;
  log_posterior: float array;
  convergence_r: float option;
}

type category_latent = {
  z_k: tensor;
  omega_k: tensor;
  z_tilde_k: tensor option;
}

type mnl_latent = {
  utilities: tensor;
  category_vars: category_latent array;
}

type mnl_params = {
  beta: tensor array;
  gamma: float array;
  delta: float array;
  prior_var: tensor;
}

type mnl_state = {
  latent: mnl_latent;
  params: mnl_params;
}

type binomial_obs = {
  y: tensor;  (* Successes *)
  n: tensor;  (* Number of trials *)
}

type binomial_latent = {
  w: tensor;
  v: tensor;
  omega_w: tensor;
  omega_v: tensor;
  w_tilde: tensor option;
  v_tilde: tensor option;
}

type bin_state = {
  latent: binomial_latent;
  params: model_params;
}

(* Numerical Utility Functions *)
let stable_log1p x =
  let x = Torch.to_float0_exn x in
  if x < -1.0 then Float.nan  
  else if abs_float x < 1e-8 then
    x -. x *. x /. 2.0  
  else log1p x

let stable_sigmoid x =
  let x = Torch.to_float0_exn x in
  if x > 15.0 then 1.0
  else if x < -15.0 then 0.0
  else 1.0 /. (1.0 +. exp(-.x))

let stable_rsqrt x =
  let eps = 1e-12 in
  Torch.(rsqrt (add x (f eps)))

let stable_softmax x =
  let max_x = Torch.max_values x ~dim:[1] in
  let shifted = Torch.(sub x (expand max_x ~size:[Torch.size x 0; -1])) in
  let exp_x = Torch.exp shifted in
  let sum_exp = Torch.sum exp_x ~dim:[1] in
  Torch.(div exp_x (expand sum_exp ~size:[Torch.size x 0; -1]))

let safe_cholesky mat =
  let n = Torch.size mat 0 in
  let jitter = 1e-6 in
  let rec try_chol eps =
    let mat_reg = Torch.(add mat (mul (eye n) (f eps))) in
    try 
      Torch.linalg.cholesky mat_reg
    with _ ->
      if eps > 1e2 then raise (Failure "Cholesky failed")
      else try_chol (eps *. 10.0)
  in
  try_chol jitter

let symmetrize mat =
  let triu = Torch.triu mat in
  let tril = Torch.tril mat ~diagonal:(-1) in
  Torch.(add triu (transpose tril 0 1))

(* Pólya-Gamma Distribution Core *)
let n_trunc = 200
let c_pg = 0.25 *. Float.pi *. Float.pi

let sample_pg1_0 () =
  let rec sum_terms k acc =
    if k >= n_trunc then acc
    else
      let d_k = Float.((2.0 *. float k +. 1.0) ** (-2.0)) in
      let g_k = Torch.exponential ~lambda:1.0 [|1|] in
      sum_terms (k+1) (acc +. d_k *. Torch.to_float0_exn g_k)
  in
  c_pg *. sum_terms 0 0.0

let sample_pg ~b ~c shape =
  let n = Torch.(numel (zeros shape)) in
  let samples = Torch.zeros shape in
  
  for i = 0 to n-1 do
    let ci = abs_float c in
    if ci < 1e-3 then begin
      let mu = b /. (2.0 *. ci) in
      let sigma = sqrt(b) /. ci in
      let z = Torch.randn [|1|] |> Torch.to_float0_exn in
      Torch.fill_
        (Torch.narrow samples ~dim:0 ~start:i ~length:1)
        (mu +. sigma *. z)
    end else if ci > 1e3 then begin
      let alpha = b /. 2.0 in
      let beta = b *. ci *. ci /. 2.0 in
      Torch.fill_
        (Torch.narrow samples ~dim:0 ~start:i ~length:1)
        (1.0 /. (Torch.gamma ~alpha ~beta [|1|] |>
                 Torch.to_float0_exn))
    end else begin
      (* Regular case *)
      Torch.fill_
        (Torch.narrow samples ~dim:0 ~start:i ~length:1)
        (sample_pg1_0 () *. exp(-0.5 *. c *. c))
    end
  done;
  samples

let sample_conditional_omega ~z =
  let abs_z = Torch.abs z in
  sample_pg ~b:2.0 ~c:(Torch.to_float0_exn abs_z) (Torch.size z)

(* Sampling Utilities *)
let sample_truncated_utility ~mu ~sigma ~lower ~upper ~max_tries =
  let rec sample_attempt n =
    if n >= max_tries then
      let rec reject_sample () =
        let z = Torch.randn [|1|] |> Torch.to_float0_exn in
        let x = mu +. sigma *. z in
        if x > lower && x < upper then x
        else reject_sample ()
      in
      reject_sample ()
    else
      let u = Torch.rand [|1|] |> Torch.to_float0_exn in
      let alpha = (lower -. mu) /. sigma in
      let beta = (upper -. mu) /. sigma in
      
      let phi_alpha = 0.5 *. (1.0 +. erf (alpha /. sqrt 2.0)) in
      let phi_beta = 0.5 *. (1.0 +. erf (beta /. sqrt 2.0)) in
      
      let z = sqrt 2.0 *. 
        erfinv (2.0 *. (phi_alpha +. u *. (phi_beta -. phi_alpha)) -. 1.0) in
        
      let x = mu +. sigma *. z in
      if x > lower && x < upper then x
      else sample_attempt (n+1)
  in
  sample_attempt 0

let sample_binary_utility ~x_beta ~y ~n =
  let utilities = Torch.zeros [|n; 1|] in
  
  for i = 0 to n-1 do
    let yi = Torch.get y [|i; 0|] |> int_of_float in
    let eta = Torch.get x_beta [|i; 0|] |> Float.of_float in
    
    let zi = if yi = 1 then
      sample_truncated_utility 
        ~mu:eta
        ~sigma:1.0
        ~lower:0.0
        ~upper:Float.infinity
        ~max_tries:100
    else
      sample_truncated_utility
        ~mu:eta
        ~sigma:1.0
        ~lower:Float.neg_infinity
        ~upper:0.0
        ~max_tries:100
    in
    
    Torch.fill_
      (Torch.narrow utilities ~dim:0 ~start:i ~length:1)
      zi
  done;
  utilities

let beta_function a b =
  let lbeta = Stdlib.log (Float.gamma a) +. 
              Stdlib.log (Float.gamma b) -. 
              Stdlib.log (Float.gamma (a +. b)) in
  exp lbeta

let log_density tau_squared a b =
  let beta = beta_function a b in
  (a -. 1.) *. log tau_squared -. 
  (a +. b) *. log (1. +. tau_squared) -. 
  log beta

let sample_gli ~nu ~loc ~scale shape =
  let samples = Torch.zeros shape in
  for i = 0 to (Torch.numel samples) - 1 do
    let u = Torch.rand [|1|] |> Torch.to_float0_exn in
    let w = Float.pow u (1.0 /. nu) in
    let gli = log w -. log(1.0 -. w) in
    Torch.fill_
      (Torch.narrow samples ~dim:0 ~start:i ~length:1)
      (loc +. scale *. gli)
  done;
  samples

let sample_glii ~nu ~loc ~scale shape =
  let samples = Torch.zeros shape in
  for i = 0 to (Torch.numel samples) - 1 do
    let u = Torch.rand [|1|] |> Torch.to_float0_exn in
    let w = Float.pow u (1.0 /. nu) in
    let glii = log w in
    Torch.fill_
      (Torch.narrow samples ~dim:0 ~start:i ~length:1)
      (loc +. scale *. glii)
  done;
  samples

(* Joint Posterior *)
module JointPosterior = struct
  type conditional_params = {
    mean: tensor;
    precision: tensor;
    bounds: float * float;
  }

  let calc_joint_log_posterior ~x ~y ~state =
    let linear_pred = Torch.(mm x state.params.beta) in
    let z = state.latent.z in
    
    (* Log likelihood from latent variables *)
    let log_lik_z = Torch.(
      sum (neg (pow (sub z linear_pred) (f 2.0))) |>
      mul (f (-0.5))
    ) |> Torch.to_float0_exn in
    
    (* Log likelihood from binary outcomes *)
    let log_lik_y = Torch.(
      sum (
        where (gt y (f 0.0))
          (log (sigmoid z))
          (log (sub (f 1.0) (sigmoid z)))
      )
    ) |> Torch.to_float0_exn in
    
    (* Log prior for beta *)
    let log_prior_beta = 
      let quad_form = Torch.(
        mm (transpose state.params.beta 0 1)
           (mm (inverse state.params.prior_var) state.params.beta)
      ) |> Torch.to_float0_exn in
      -0.5 *. quad_form in
      
    (* Log prior for expansion parameters *)
    let log_prior_gamma = 
      -0.5 *. state.params.gamma *. state.params.gamma in
    let log_prior_delta =
      -2.0 *. log(state.params.delta) in

    log_lik_z +. log_lik_y +. log_prior_beta +. 
    log_prior_gamma +. log_prior_delta

  let get_conditional_beta ~x ~z ~omega ~prior_var =
    let n = Torch.size x 0 in
    let d = Torch.size x 1 in
    
    (* Precision matrix *)
    let x_omega = Torch.(mul x (expand omega ~size:[n; d])) in
    let precision = Torch.(
      add (inverse prior_var) 
          (mm (transpose x_omega 0 1) x)
    ) in
    
    (* Mean vector *)
    let omega_z = Torch.(mul omega z) in
    let mean = Torch.(
      mm (inverse precision) 
         (mm (transpose x 0 1) omega_z)
    ) in
    
    { mean; precision; bounds = (Float.neg_infinity, Float.infinity) }

  let get_conditional_gamma ~z_tilde ~y =
    let n = Torch.size y 0 in
    let zeros = Torch.zeros [|n; 1|] in
    
    let l_gamma = Torch.(
      where (eq y zeros) z_tilde (neg (f Float.infinity))
    ) |> Torch.max_values ~dim:[0] |> Torch.to_float0_exn in
    
    let u_gamma = Torch.(
      where (gt y zeros) z_tilde (f Float.infinity)  
    ) |> Torch.min_values ~dim:[0] |> Torch.to_float0_exn in
    
    let precision = Torch.float_value 1.0 in
    let mean = Torch.float_value 0.0 in
    
    { mean; precision; bounds = (l_gamma, u_gamma) }

  let get_conditional_delta ~z ~omega =
    let n = Torch.size z 0 in
    
    (* Shape and rate for inverse gamma *)
    let alpha = float n /. 2.0 +. 2.0 in
    let beta = Torch.(
      sum (mul omega (pow z (f 2.0))) |> 
      div (f 2.0) |>  
      add (f 2.0)
    ) |> Torch.to_float0_exn in
    
    let precision = Torch.float_value (1.0 /. beta) in
    let mean = Torch.float_value (1.0 /. (alpha -. 1.0) *. beta) in
    
    { mean; precision; bounds = (0.0, Float.infinity) }
end

(* Parameter Expansion *)
let expand_location ~z ~gamma ~y =
  let z_tilde = Torch.(add z (f gamma)) in
  let pos_mask = Torch.(gt y (f 0.0)) in
  let neg_mask = Torch.(eq y (f 0.0)) in
  
  let l_gamma = Torch.(
    where neg_mask z_tilde (neg (f Float.infinity)) |>
    max_values ~dim:[0]
  ) in
  
  let u_gamma = Torch.(
    where pos_mask z_tilde (f Float.infinity) |>
    min_values ~dim:[0]
  ) in
  
  z_tilde, l_gamma, u_gamma

let expand_scale ~z ~delta ~omega =
  let eps = 1e-10 in
  let sqrt_delta = sqrt (delta +. eps) in
  let z_scaled = Torch.(mul z (f sqrt_delta)) in
  let omega_scaled = Torch.(div omega (add (f delta) (f eps))) in
  z_scaled, omega_scaled

let expand_parameters ~state ~y =
  (* Location expansion *)
  let z_tilde, l_gamma, u_gamma = 
    expand_location ~z:state.latent.z ~gamma:state.params.gamma ~y in
  
  (* Scale expansion *)
  let z_scaled, omega_scaled = 
    expand_scale ~z:z_tilde ~delta:state.params.delta ~omega:state.latent.omega in
  
  { state with
    latent = { state.latent with
              z = z_scaled;
              omega = omega_scaled;
              z_tilde = Some z_tilde };
    params = state.params }

(* MCMC Kernel *)
module MCMCKernel = struct
  type kernel_stats = {
    log_posterior: float;
    acceptance_rate: float;
    step_size: float;
  }

  let transition ~state ~x ~y =
    try
      (* Sample latent utilities *)
      let z = sample_binary_utility
        ~x_beta:Torch.(mm x state.params.beta)
        ~y
        ~n:(Torch.size x 0) in

      (* Sample auxiliary variables *)
      let omega = sample_conditional_omega ~z in

      let state = { state with 
                    latent = { state.latent with
                             z; omega;
                             z_tilde = None } } in

      (* Parameter expansion *)
      let z_tilde, l_gamma, u_gamma = expand_location 
        ~z:state.latent.z
        ~gamma:state.params.gamma
        ~y in

      (* Sample new gamma *)
      let gamma_new = sample_truncated_utility
        ~mu:0.0
        ~sigma:1.0
        ~lower:(Torch.to_float0_exn l_gamma)
        ~upper:(Torch.to_float0_exn u_gamma)
        ~max_tries:100 in

      (* Scale expansion *)
      let z_scaled, omega_scaled = expand_scale
        ~z:z_tilde
        ~delta:state.params.delta
        ~omega in

      (* Sample new delta *)
      let n = Torch.size x 0 in
      let alpha_n = float n /. 2.0 +. 2.0 in
      let beta_n = Torch.(
        sum (mul omega_scaled (pow z_scaled (f 2.0))) |>
        div (f 2.0) |>
        add (f 2.0)
      ) |> Torch.to_float0_exn in

      let delta_new = 1.0 /. (Torch.gamma ~alpha:alpha_n ~beta:beta_n [|1|] |>
                             Torch.to_float0_exn) in

      (* Sample beta *)
      let beta_params = JointPosterior.get_conditional_beta
        ~x ~z:z_scaled
        ~omega:omega_scaled
        ~prior_var:state.params.prior_var in

      let beta = let eps = Torch.randn [|Torch.size beta_params.mean 0; 1|] in
                 let chol = safe_cholesky beta_params.precision in
                 Torch.(add beta_params.mean (mm (inverse chol) eps)) in

      let new_state = { params = { state.params with
                                  beta;
                                  gamma = gamma_new;
                                  delta = delta_new };
                       latent = { state.latent with
                                z = z_scaled;
                                omega = omega_scaled;
                                z_tilde = Some z_tilde } } in

      (* Calculate statistics *)
      let log_post = JointPosterior.calc_joint_log_posterior ~x ~y ~state:new_state in
      let old_log_post = JointPosterior.calc_joint_log_posterior ~x ~y ~state in
      let acc_rate = if log_post > old_log_post then 1.0 else 0.0 in

      Ok (new_state, { log_posterior = log_post;
                      acceptance_rate = acc_rate;
                      step_size = delta_new })

    with e ->
      Error (Printf.sprintf "MCMC transition failed: %s" (Printexc.to_string e))
end

let calc_offset ~x ~beta ~k ~n_categories =
  let n = Torch.size x 0 in
  let sum_exp = Torch.zeros [|n; 1|] in
  
  (* Sum exp(x_i'β_l) over l ≠ k *)
  for l = 0 to n_categories-1 do
    if l <> k then
      let x_beta_l = Torch.(mm x beta.(l)) in
      let exp_l = Torch.(exp (neg x_beta_l)) in
      Torch.(sum_exp += exp_l)
  done;
  
  Torch.(log1p sum_exp)

let sample_utilities ~x ~beta ~y ~n_categories =
  let n = Torch.size x 0 in
  let utilities = Torch.zeros [|n; n_categories|] in
  
  (* Sample u_i ~ EV(x_i'β_k) *)
  for k = 0 to n_categories-1 do
    if k > 0 then begin
      let x_beta = Torch.(mm x beta.(k)) in
      let u = Torch.rand [|n; 1|] in
      
      let util_k = Torch.(
        sub x_beta (log (neg (log u)))
      ) in
      
      Torch.copy_ 
        (Torch.select utilities ~dim:1 ~index:k)
        util_k
    end
  done;
  utilities

let calc_utility_gaps ~utilities ~n_categories =
  let n = Torch.size utilities 0 in
  let gaps = Array.make n_categories (Torch.zeros [|n; 1|]) in
  
  for k = 1 to n_categories-1 do
    let u_k = Torch.select utilities ~dim:1 ~index:k in
    
    (* Get max among other categories *)
    let mask = Torch.ones [|n_categories|] in
    Torch.fill_ (Torch.select mask ~dim:0 ~index:k) 0.0;
    
    let u_others = Torch.(
      masked_select utilities (expand mask ~size:[n; n_categories])
    ) in
    let max_others = Torch.max_values u_others ~dim:[1] in
    
    gaps.(k) <- Torch.(sub u_k max_others)
  done;
  gaps

(* Binomial Sampling *)
let sample_dual_latents ~x ~beta ~obs =
  let n = Torch.size x 0 in
  let linear_pred = Torch.(mm x beta) in
  
  (* Pre-allocate tensors *)
  let w = Torch.zeros [|n; 1|] in
  let v = Torch.zeros [|n; 1|] in
  let omega_w = Torch.zeros [|n; 1|] in
  let omega_v = Torch.zeros [|n; 1|] in
  
  (* Vectorized sampling *)
  for i = 0 to n-1 do
    let yi = Torch.get obs.y [|i; 0|] |> int_of_float in
    let ni = Torch.get obs.n [|i; 0|] |> int_of_float in
    let eta_i = Torch.get linear_pred [|i; 0|] |> Float.of_float in
    
    if yi > 0 then begin
      (* Sample w *)
      let nu = float yi in
      let w_i = sample_glii ~nu ~loc:eta_i ~scale:1.0 [|1; 1|] in
      Torch.copy_ 
        (Torch.narrow w ~dim:0 ~start:i ~length:1)
        w_i;
        
      let omega_wi = sample_pg
        ~b:(nu +. 1.0)
        ~c:(Torch.to_float0_exn w_i)
        [|1|] in
      Torch.copy_
        (Torch.narrow omega_w ~dim:0 ~start:i ~length:1)
        omega_wi
    end;
    
    if yi < ni then begin 
      (* Sample v *)
      let nu = float (ni - yi) in
      let v_i = sample_gli ~nu ~loc:eta_i ~scale:1.0 [|1; 1|] in
      Torch.copy_
        (Torch.narrow v ~dim:0 ~start:i ~length:1)
        v_i;
        
      let omega_vi = sample_pg
        ~b:(nu +. 1.0)
        ~c:(Torch.to_float0_exn v_i)
        [|1|] in
      Torch.copy_
        (Torch.narrow omega_v ~dim:0 ~start:i ~length:1)
        omega_vi
    end
  done;
  
  w, v, omega_w, omega_v

(* Utility Expansion *)
let expand_utilities ~w ~v ~omega_w ~omega_v ~gamma ~delta ~obs =
  (* Location expansion *)
  let w_tilde = Torch.(add w (f gamma)) in
  let v_tilde = Torch.(add v (f gamma)) in
  
  (* Calculate bounds *)
  let bounds = ref [] in
  let n = Torch.size obs.y 0 in
  
  for i = 0 to n-1 do
    let yi = Torch.get obs.y [|i; 0|] |> int_of_float in
    let ni = Torch.get obs.n [|i; 0|] |> int_of_float in
    let wi = if yi > 0 then
      Torch.get w_tilde [|i; 0|] |> Torch.to_float0_exn
    else Float.neg_infinity in
    let vi = if yi < ni then
      Torch.get v_tilde [|i; 0|] |> Torch.to_float0_exn  
    else Float.infinity in
    
    bounds := (wi, vi) :: !bounds
  done;
  
  (* Get overall bounds *)
  let l_gamma = List.fold_left (fun acc (w,_) -> max acc w) 
                  Float.neg_infinity !bounds in
  let u_gamma = List.fold_left (fun acc (_,v) -> min acc v) 
                  Float.infinity !bounds in
  
  (* Scale expansion *)
  let sqrt_delta = sqrt delta in
  let w_scaled = Torch.(mul w_tilde (f sqrt_delta)) in
  let v_scaled = Torch.(mul v_tilde (f sqrt_delta)) in
  let omega_w_scaled = Torch.(div omega_w (f delta)) in
  let omega_v_scaled = Torch.(div omega_v (f delta)) in
  
  (w_tilde, v_tilde),
  (w_scaled, v_scaled),
  (omega_w_scaled, omega_v_scaled),
  (l_gamma, u_gamma)

(* MCMC Engine *)
module MCMCEngine = struct
  let initialize ~x ~y =
    let n = Torch.size x 0 in
    let d = Torch.size x 1 in
    
    (* Initial regression using probit approximation *)
    let y_std = Torch.(div (sub y (mean y)) (std y)) in
    let x_std = Torch.(
      div (sub x (mean x ~dim:[0] ~keepdim:true))
          (std x ~dim:[0] ~keepdim:true)
    ) in
    
    (* OLS estimate *)
    let beta_init = Torch.(
      mm (mm (inverse (mm (transpose x_std 0 1) x_std))
             (transpose x_std 0 1))
         y_std
    ) in
    
    (* Initial latent utilities *)
    let z_init = Torch.(mm x beta_init) in
    let omega_init = Torch.ones [|n; 1|] in
    
    (* Prior variance *)
    let prior_var = Torch.(mul (eye d) (f 100.0)) in
    
    { params = {
        beta = beta_init;
        gamma = 0.0;
        delta = 1.0;
        prior_var = prior_var
      };
      latent = {
        z = z_init;
        omega = omega_init;
        z_tilde = None
      } }

  let calc_ess samples =
    let n = List.length samples in
    if n < 2 then 1.0
    else
      let mean = List.fold_left (+.) 0.0 samples /. float n in
      
      (* Calculate autocorrelations *)
      let max_lag = min 100 (n/2) in
      let auto_corr = Array.make max_lag 0.0 in
      
      for lag = 0 to max_lag-1 do
        let sum_prod = ref 0.0 in
        let sum_sq = ref 0.0 in
        for i = 0 to n-lag-1 do
          let x_i = List.nth samples i -. mean in
          let x_lag = List.nth samples (i+lag) -. mean in
          sum_prod := !sum_prod +. x_i *. x_lag;
          sum_sq := !sum_sq +. x_i *. x_i
        done;
        auto_corr.(lag) <- !sum_prod /. !sum_sq
      done;
      
      (* Calculate ESS *)
      let tau = 1.0 +. 2.0 *. (Array.fold_left (+.) 0.0 auto_corr) in
      float n /. tau

  let calc_rhat ~chains =
    let n_chains = List.length chains in
    let chain_length = List.length (List.hd chains) in
    
    (* Calculate chain means *)
    let chain_means = List.map (fun chain ->
      List.fold_left (fun acc state ->
        acc +. (Torch.to_float0_exn state.params.beta)
      ) 0.0 chain /. float chain_length
    ) chains in
    
    (* Overall mean *)
    let overall_mean = 
      List.fold_left (+.) 0.0 chain_means /. float n_chains in
    
    (* Between-chain variance *)
    let b = List.fold_left (fun acc mu ->
      acc +. (mu -. overall_mean) ** 2.0
    ) 0.0 chain_means *. float chain_length /. float (n_chains - 1) in
    
    (* Within-chain variance *)
    (* Within-chain variance cont'd *)
    let w = List.fold_left (fun acc chain ->
      let chain_mean = List.hd chain_means in
      acc +. List.fold_left (fun acc2 state ->
        let x = Torch.to_float0_exn state.params.beta in
        acc2 +. (x -. chain_mean) ** 2.0
      ) 0.0 chain
    ) 0.0 chains /. float (n_chains * (chain_length - 1)) in
    
    (* Calculate R-hat *)
    sqrt ((float (chain_length - 1) /. float chain_length *. w +. b/. float chain_length) /. w)
end

module Diagnostics = struct
  type sampler_stats = {
    acceptance_rates: float array;
    effective_samples: float array;
    r_hat: float option;
    runtime: float;
  }

  let calc_acceptance_rates chain =
    let n_params = match chain with
      | [] -> 0
      | state::_ -> Torch.size state.params.beta 0
    in
    let acc = Array.make n_params 0 in
    
    List.iteri (fun i state ->
      if i > 0 then
        let prev = List.nth chain (i-1) in
        for j = 0 to n_params-1 do
          let diff = Torch.(
            abs (sub 
              (narrow state.params.beta ~dim:0 ~start:j ~length:1)
              (narrow prev.params.beta ~dim:0 ~start:j ~length:1))
          ) in
          if Torch.to_float0_exn diff > 1e-8 then
            acc.(j) <- acc.(j) + 1
        done
    ) chain;
    
    Array.map (fun a -> float a /. float (List.length chain - 1)) acc
end

module UPGG = struct
  type model = 
    | Binary 
    | Multinomial of int
    | Binomial

  type config = {
    n_warmup: int;
    n_iter: int;
    n_chains: int;
    target_acceptance: float;
    rare_threshold: float;
  }

  let default_config = {
    n_warmup = 1000;
    n_iter = 5000;
    n_chains = 4;
    target_acceptance = 0.234;
    rare_threshold = 0.05
  }

  let sample_binary ~config ~x ~y =
    let t_start = Unix.gettimeofday() in
    
    (* Initialize chains *)
    let chains = List.init config.n_chains (fun _ ->
      MCMCEngine.initialize ~x ~y
    ) in
    
    (* Run chains *)
    let results = List.map (fun init_state ->
      let rec run i state chain =
        if i >= config.n_iter then
          List.rev chain
        else
          match MCMCKernel.transition ~state ~x ~y with
          | Ok (state', _) -> run (i+1) state' (state' :: chain)
          | Error msg ->
            Printf.printf "Warning: %s\n" msg;
            run (i+1) state chain
      in
      run 0 init_state []
    ) chains in
    
    let runtime = Unix.gettimeofday() -. t_start in
    
    (* Calculate diagnostics *)
    let r_hat = 
      if config.n_chains > 1 then
        Some (MCMCEngine.calc_rhat ~chains:results)
      else None in
      
    let acc_rates = List.map (fun chain ->
      Diagnostics.calc_acceptance_rates chain
    ) results in
    
    let eff_samples = List.map (fun chain ->
      Array.init (Torch.size (List.hd chain).params.beta 0) (fun i ->
        let param_samples = List.map (fun state ->
          Torch.to_float0_exn 
            (Torch.narrow state.params.beta ~dim:0 ~start:i ~length:1)
        ) chain in
        MCMCEngine.calc_ess param_samples
      )
    ) results in
    
    results,
    { Diagnostics.
      acceptance_rates = Array.of_list acc_rates;
      effective_samples = Array.of_list eff_samples;
      r_hat;
      runtime
    }

  let sample_multinomial ~config ~x ~y ~n_categories =
    let t_start = Unix.gettimeofday() in
    
    (* Initialize chains *)
    let chains = List.init config.n_chains (fun _ ->
      let n = Torch.size x 0 in
      let d = Torch.size x 1 in
      
      (* Initial beta parameters *)
      let beta_init = Array.init n_categories (fun _ ->
        let x_std = Torch.(
          div (sub x (mean x ~dim:[0] ~keepdim:true))
              (std x ~dim:[0] ~keepdim:true)
        ) in
        let y_k = Torch.zeros [|n; 1|] in
        let xtx = Torch.(mm (transpose x_std 0 1) x_std) in
        let xty = Torch.(mm (transpose x_std 0 1) y_k) in
        Torch.(mm (inverse xtx) xty)
      ) in
      
      (* Initial latent variables *)
      let category_vars = Array.init n_categories (fun k ->
        let z_k = Torch.(mm x beta_init.(k)) in
        let omega_k = Torch.ones [|n; 1|] in
        { MultinomialTypes.
          z_k;
          omega_k;
          z_tilde_k = None }
      ) in
      
      { MultinomialTypes.
        latent = {
          utilities = Torch.zeros [|n; n_categories|];
          category_vars;
        };
        params = {
          beta = beta_init;
          gamma = Array.make n_categories 0.0;
          delta = Array.make n_categories 1.0;
          prior_var = Torch.(mul (eye d) (f 100.0));
        }
      }
    ) in
    
    (* Run chains *)
    let results = List.map (fun init_state ->
      let rec run i state chain =
        if i >= config.n_iter then
          List.rev chain
        else begin
          (* Sample utilities *)
          let utilities = MultinomialSampling.sample_utilities
            ~x ~beta:state.params.beta ~y ~n_categories in
          
          (* Sample category-specific variables *)
          let category_vars = Array.init n_categories (fun k ->
            let u_k = Torch.select utilities ~dim:1 ~index:k in
            let omega_k = sample_conditional_omega ~z:u_k in
            
            (* Location-scale expansion *)
            let z_tilde_k, l_k, u_k = expand_location 
              ~z:u_k 
              ~gamma:state.params.gamma.(k)
              ~y in
            
            let z_k_scaled, omega_k_scaled = expand_scale
              ~z:z_tilde_k
              ~delta:state.params.delta.(k)
              ~omega:omega_k in
              
            { MultinomialTypes.
              z_k = z_k_scaled;
              omega_k = omega_k_scaled;
              z_tilde_k = Some z_tilde_k }
          ) in
          
          (* Update parameters *)
          let new_betas = Array.init n_categories (fun k ->
            let beta_params = JointPosterior.get_conditional_beta
              ~x 
              ~z:category_vars.(k).z_k
              ~omega:category_vars.(k).omega_k
              ~prior_var:state.params.prior_var in
              
            let eps = Torch.randn [|Torch.size beta_params.mean 0; 1|] in
            let chol = safe_cholesky beta_params.precision in
            Torch.(add beta_params.mean (mm (inverse chol) eps))
          ) in
          
          let new_state = { MultinomialTypes.
            latent = {
              utilities;
              category_vars;
            };
            params = { state.params with beta = new_betas }
          } in
          
          run (i+1) new_state (state :: chain)
        end
      in
      run 0 init_state []
    ) chains in
    
    let runtime = Unix.gettimeofday() -. t_start in
    
    (* Calculate diagnostics *)
    let r_hat = 
      if config.n_chains > 1 then
        Some (MCMCEngine.calc_rhat ~chains:results)
      else None in
      
    let acc_rates = List.map (fun chain ->
      Diagnostics.calc_acceptance_rates chain
    ) results in
    
    results,
    { Diagnostics.
      acceptance_rates = Array.of_list acc_rates;
      effective_samples = [||]; (* TODO: Implement for multinomial *)
      r_hat;
      runtime }

  let sample_binomial ~config ~x ~y ~n =
    let t_start = Unix.gettimeofday() in
    
    let obs = { BinomialTypes.y; n } in
    
    (* Initialize chains *)
    let chains = List.init config.n_chains (fun _ ->
      MCMCEngine.initialize ~x ~y
    ) in
    
    (* Run chains *)
    let results = List.map (fun init_state ->
      let rec run i state chain =
        if i >= config.n_iter then
          List.rev chain
        else begin
          (* Sample latent variables *)
          let (w, v, omega_w, omega_v) = BinomialSampling.sample_dual_latents
            ~x ~beta:state.params.beta ~obs in
            
          (* Expand utilities *)
          let (w_tilde, v_tilde), 
              (w_scaled, v_scaled),
              (omega_w_scaled, omega_v_scaled),
              (l_gamma, u_gamma) =
            expand_utilities 
              ~w ~v ~omega_w ~omega_v
              ~gamma:state.params.gamma
              ~delta:state.params.delta
              ~obs in
              
          (* Update gamma using bounds *)
          let gamma_new = sample_truncated_utility
            ~mu:0.0
            ~sigma:1.0
            ~lower:l_gamma
            ~upper:u_gamma
            ~max_tries:100 in
            
          (* Update delta *)
          let n_obs = Torch.size obs.y 0 in
          let alpha_n = float n_obs /. 2.0 +. 2.0 in
          let beta_n = Torch.(
            add (add
              (sum (mul omega_w_scaled (pow w_scaled (f 2.0))))
              (sum (mul omega_v_scaled (pow v_scaled (f 2.0)))))
              (f 2.0)
          ) |> Torch.to_float0_exn in
          
          let delta_new = 1.0 /. (Torch.gamma ~alpha:alpha_n ~beta:beta_n [|1|] |>
                                 Torch.to_float0_exn) in
                                 
          (* Update beta *)
          let z = Torch.cat [w_scaled; v_scaled] ~dim:0 in
          let omega = Torch.cat [omega_w_scaled; omega_v_scaled] ~dim:0 in
          
          let beta_params = JointPosterior.get_conditional_beta
            ~x ~z ~omega ~prior_var:state.params.prior_var in
            
          let beta = let eps = Torch.randn [|Torch.size beta_params.mean 0; 1|] in
                     let chol = safe_cholesky beta_params.precision in
                     Torch.(add beta_params.mean (mm (inverse chol) eps)) in
                     
          let new_state = { BinomialTypes.
            latent = {
              w; v;
              omega_w; omega_v;
              w_tilde = Some w_tilde;
              v_tilde = Some v_tilde;
            };
            params = {
              state.params with
              beta;
              gamma = gamma_new;
              delta = delta_new;
            }
          } in
          
          run (i+1) new_state (state :: chain)
        end
      in
      run 0 init_state []
    ) chains in
    
    let runtime = Unix.gettimeofday() -. t_start in
    
    results,
    { Diagnostics.
      acceptance_rates = Array.make 1 0.0; (* TODO: Proper acceptance tracking *)
      effective_samples = [||];
      r_hat = None;
      runtime }

  let sample ?config ~x ~y model =
    let config = match config with
      | Some c -> c
      | None -> default_config
    in
    match model with
    | Binary -> sample_binary ~config ~x ~y
    | Multinomial n -> sample_multinomial ~config ~x ~y ~n_categories:n
    | Binomial -> 
        let n = Torch.ones (Torch.size y) in
        sample_binomial ~config ~x ~y ~n
end