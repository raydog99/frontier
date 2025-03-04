open Torch

let log_gamma x =
  let pi = Scalar.float Float.pi in
  let rec g z =
    if Scalar.(z < float 7.0) then
      g Scalar.(z + float 1.0) - Scalar.log z
    else
      let x = Scalar.(float 1.0 / z) in
      Scalar.(log z - float 1.0 / z + float 0.5 * log (float 2.0 * pi / z) +
      float 1.0 / (float 12.0 * z) -
      float 1.0 / (float 120.0 * z * z * z * z * z) +
      float 1.0 / (float 252.0 * z * z * z * z * z * z * z))
  in
  if Scalar.(x <= float 0.0) then
    failwith "log_gamma: x must be positive"
  else
    g x

let digamma x =
  let rec h x =
    if Scalar.(x >= float 8.0) then
      Scalar.(log x - float 0.5 / x - float 1.0 / (float 12.0 * x * x) + 
             float 1.0 / (float 120.0 * x * x * x * x) -
             float 1.0 / (float 252.0 * x * x * x * x * x * x))
    else
      h Scalar.(x + float 1.0) - Scalar.(float 1.0 / x)
  in
  h x

(* Transform constrained GARCH parameters to unconstrained space *)
let to_unconstrained ~omega ~alpha ~beta =
  let omega_u = Tensor.log omega in
  let psi1 = Tensor.div alpha Tensor.(alpha + beta) in
  let psi2 = Tensor.div beta Tensor.(alpha + beta) in
  let psi1_u = Tensor.(log (psi1 / (ones_like psi1 - psi1))) in
  let psi2_u = Tensor.(log (psi2 / (ones_like psi2 - psi2))) in
  (omega_u, psi1_u, psi2_u)

(* Transform unconstrained parameters back to constrained GARCH space *)
let to_constrained ~omega_u ~psi1_u ~psi2_u =
  let omega = Tensor.exp omega_u in
  let psi1 = Tensor.(exp psi1_u / (ones_like psi1_u + exp psi1_u)) in
  let psi2 = Tensor.(exp psi2_u / (ones_like psi2_u + exp psi2_u)) in 
  let alpha = Tensor.(psi1 * (ones_like psi1 - psi2)) in
  let beta = Tensor.(psi2 * (ones_like psi2 - psi1)) in
  (omega, alpha, beta)
  
(* Transform degrees of freedom parameter *)
let transform_nu_to_unconstrained nu =
  Tensor.log Tensor.(nu - scalar (float 2.0))
  
let transform_nu_from_unconstrained nu_u =
  Tensor.(exp nu_u + scalar (float 2.0))
  
(* Transform skewness parameter *)
let transform_xi_to_unconstrained xi =
  Tensor.log xi
  
let transform_xi_from_unconstrained xi_u =
  Tensor.exp xi_u

let garch_1_1 ~omega ~alpha ~beta ~y =
  let t = Tensor.shape y |> List.hd in
  let sigma2 = Tensor.zeros [t] in
  
  (* Initialize with unconditional variance *)
  let unconditional_var = Tensor.(omega / (ones_like omega - alpha - beta)) in
  Tensor.copy_ ~src:(Tensor.repeat unconditional_var ~repeats:[1]) ~dst:(Tensor.slice sigma2 ~dim:0 ~start:0 ~end_:1 ~step:1);
  
  (* Compute GARCH process recursively *)
  for i = 1 to t - 1 do
    let prev_sigma2 = Tensor.get sigma2 (i - 1) |> Tensor.unsqueeze ~dim:0 in
    let prev_y = Tensor.get y (i - 1) |> Tensor.unsqueeze ~dim:0 in
    let prev_y2 = Tensor.pow prev_y (Tensor.of_float0 2.0) in
    let new_sigma2 = Tensor.(omega + alpha * prev_y2 + beta * prev_sigma2) in
    Tensor.copy_ ~src:new_sigma2 ~dst:(Tensor.slice sigma2 ~dim:0 ~start:i ~end_:(i+1) ~step:1);
  done;
  
  sigma2

(* Compute partial derivatives of sigma2 with respect to unconstrained parameters *)
let d_sigma2_d_theta_garch_1_1 ~omega ~alpha ~beta ~omega_u ~psi1_u ~psi2_u ~y =
  let t = Tensor.shape y |> List.hd in
  let d_sigma2_d_omega = Tensor.ones [t] in
  let d_sigma2_d_alpha = Tensor.zeros [t] in
  let d_sigma2_d_beta = Tensor.zeros [t] in
  
  (* Compute derivatives recursively *)
  let y2 = Tensor.(pow y (scalar (float 2.0))) in
  
  (* Initialize first value based on stationarity *)
  let d_omega_u = Tensor.(omega) in
  let d_psi1_u = Tensor.(
    alpha * (scalar (float 1.0) - beta) *
    exp psi1_u / pow (scalar (float 1.0) + exp psi1_u) (scalar (float 2.0))
  ) in
  let d_psi2_u = Tensor.(
    beta * (scalar (float 1.0) - alpha) *
    exp psi2_u / pow (scalar (float 1.0) + exp psi2_u) (scalar (float 2.0))
  ) in
  
  Tensor.copy_ ~src:d_omega_u ~dst:(Tensor.slice d_sigma2_d_omega ~dim:0 ~start:0 ~end_:1 ~step:1);
  Tensor.copy_ ~src:d_psi1_u ~dst:(Tensor.slice d_sigma2_d_alpha ~dim:0 ~start:0 ~end_:1 ~step:1);
  Tensor.copy_ ~src:d_psi2_u ~dst:(Tensor.slice d_sigma2_d_beta ~dim:0 ~start:0 ~end_:1 ~step:1);
  
  (* Recursively compute derivatives *)
  for i = 1 to t - 1 do
    let prev_y2 = Tensor.get y2 (i-1) |> Tensor.unsqueeze ~dim:0 in
    let prev_d_omega = Tensor.get d_sigma2_d_omega (i-1) |> Tensor.unsqueeze ~dim:0 in
    let prev_d_alpha = Tensor.get d_sigma2_d_alpha (i-1) |> Tensor.unsqueeze ~dim:0 in
    let prev_d_beta = Tensor.get d_sigma2_d_beta (i-1) |> Tensor.unsqueeze ~dim:0 in
    
    let new_d_omega = Tensor.(scalar (float 1.0) + beta * prev_d_omega) in
    let new_d_alpha = Tensor.(prev_y2 + beta * prev_d_alpha) in
    let new_d_beta = Tensor.(prev_d_beta * beta + (get d_sigma2_d_omega (i-1)) |> unsqueeze ~dim:0) in
    
    Tensor.copy_ ~src:new_d_omega ~dst:(Tensor.slice d_sigma2_d_omega ~dim:0 ~start:i ~end_:(i+1) ~step:1);
    Tensor.copy_ ~src:new_d_alpha ~dst:(Tensor.slice d_sigma2_d_alpha ~dim:0 ~start:i ~end_:(i+1) ~step:1);
    Tensor.copy_ ~src:new_d_beta ~dst:(Tensor.slice d_sigma2_d_beta ~dim:0 ~start:i ~end_:(i+1) ~step:1);
  done;
  
  Tensor.stack [d_sigma2_d_omega; d_sigma2_d_alpha; d_sigma2_d_beta] ~dim:1

(* Priors for GARCH models *)
module Prior = struct
  type t = {
    log_pdf: Tensor.t -> Tensor.t;
    grad_log_pdf: Tensor.t -> Tensor.t;
  }
  
  (* Log-gamma prior for omega *)
  let neg_log_gamma ~shape ~rate =
    let log_pdf theta =
      Tensor.((shape - scalar (float 1.0)) * (log theta) - rate * theta)
    in
    let grad_log_pdf theta =
      Tensor.((shape - scalar (float 1.0)) / theta - rate)
    in
    { log_pdf; grad_log_pdf }
  
  (* Logistic prior for psi1 and psi2 *)
  let logistic () =
    let log_pdf theta =
      Tensor.(theta - scalar (float 2.0) * log (scalar (float 1.0) + exp theta))
    in
    let grad_log_pdf theta =
      Tensor.(scalar (float 1.0) - scalar (float 2.0) * exp theta / (scalar (float 1.0) + exp theta))
    in
    { log_pdf; grad_log_pdf }
    
  (* Translated exponential prior for degrees of freedom *)
  let translated_exp ~rate =
    let log_pdf theta =
      Tensor.(-(exp theta + scalar (float 1.0)) - log (scalar (float 1.0) + exp theta))
    in
    let grad_log_pdf theta =
      Tensor.(-(exp theta) / (scalar (float 1.0) + exp theta) - exp theta / (scalar (float 1.0) + exp theta))
    in
    { log_pdf; grad_log_pdf }
    
  (* Inverse Gamma prior for skewness parameter *)
  let inverse_gamma ~shape ~rate =
    let log_pdf theta =
      Tensor.(-scalar (float 2.0) * log theta - rate / theta - log (scalar (float 1.0) + exp theta))
    in
    let grad_log_pdf theta =
      Tensor.(-(scalar (float 2.0)) / theta + rate / (theta * theta) - exp theta / (scalar (float 1.0) + exp theta))
    in
    { log_pdf; grad_log_pdf }
end

(* Gaussian GARCH model log-likelihood *)
let gaussian ~y ~sigma2 =
  let t = Tensor.shape y |> List.hd in
  let t_float = Float.of_int t in
  let pi = Scalar.float Float.pi in
  
  let term1 = Tensor.(scalar (float (-0.5) *. t_float) * log (scalar (float 2.0) * scalar pi)) in
  let term2 = Tensor.(scalar (float (-0.5)) * sum (log sigma2)) in
  let term3 = Tensor.(scalar (float (-0.5)) * sum (pow y (scalar (float 2.0)) / sigma2)) in
  
  Tensor.(term1 + term2 + term3)
  
(* Gradient of Gaussian GARCH log-likelihood w.r.t. unconstrained params *)
let grad_gaussian ~y ~sigma2 ~d_sigma2_d_theta =
  let t = Tensor.shape y |> List.hd in
  let y2 = Tensor.(pow y (scalar (float 2.0))) in
  
  (* d_ll/d_sigma2 *)
  let d_ll_d_sigma2 = Tensor.(scalar (float (-0.5)) * (ones [t] - y2 / pow sigma2 (scalar (float 2.0)))) in
  
  (* d_ll/d_theta = d_ll/d_sigma2 * d_sigma2/d_theta *)
  Tensor.(d_ll_d_sigma2 * d_sigma2_d_theta)
  
(* Student's t GARCH model log-likelihood *)
let student_t ~y ~sigma2 ~nu =
  let t = Tensor.shape y |> List.hd in
  let t_float = Float.of_int t in
  let pi = Scalar.float Float.pi in
  
  let nu_minus_2 = Tensor.(nu - scalar (float 2.0)) in
  
  (* Log-gamma terms *)
  let term1 = Tensor.(scalar (float t_float) * (
                      log_gamma ((nu + scalar (float 1.0)) / scalar (float 2.0)) - 
                      log_gamma (nu / scalar (float 2.0)))) in
  
  (* Pi and variance terms *)
  let term2 = Tensor.(scalar (float (-0.5)) * scalar (float t_float) * 
                      log (nu_minus_2 * scalar pi)) in
  
  (* Sigma2 term *)
  let term3 = Tensor.(scalar (float (-0.5)) * sum (log sigma2)) in
  
  (* (1 + y^2/(nu-2)*sigma2)^(-(nu+1)/2) term *)
  let term4 = Tensor.(scalar (float (-0.5)) * (nu + scalar (float 1.0)) * 
                      sum (log (scalar (float 1.0) + pow y (scalar (float 2.0)) / 
                      (nu_minus_2 * sigma2)))) in
  
  Tensor.(term1 + term2 + term3 + term4)
  
(* Skew-t GARCH model log-likelihood *)
let skew_t ~y ~sigma2 ~nu ~xi =
  let t = Tensor.shape y |> List.hd in
  let t_float = Float.of_int t in
  let pi = Scalar.float Float.pi in
  
  let nu_minus_2 = Tensor.(nu - scalar (float 2.0)) in
  
  (* Mean and scaling parameters for skew-t *)
  let m = Tensor.(((nu_minus_2 / scalar pi) |> sqrt) * 
                  (xi - scalar (float 1.0) / xi) * 
                  gamma ((nu - scalar (float 1.0)) / scalar (float 2.0)) / 
                  gamma (nu / scalar (float 2.0))) in
  
  let s = Tensor.(sqrt (pow xi (scalar (float 2.0)) + 
                  pow (scalar (float 1.0) / xi) (scalar (float 2.0)) - 
                  scalar (float 1.0) - pow m (scalar (float 2.0)))) in
  
  (* Indicators for skewness *)
  let standardized = Tensor.(y / sqrt sigma2) in
  let indicators = Tensor.(standardized <=  m / s) in
  let xi_ind = Tensor.where indicators 
                 ~self:(Tensor.ones_like xi)
                 ~other:(Tensor.pow xi (scalar (float 2.0)));
  
  (* Log-gamma terms *)
  let term1 = Tensor.(scalar (float t_float) * (
                      log_gamma ((nu + scalar (float 1.0)) / scalar (float 2.0)) - 
                      log_gamma (nu / scalar (float 2.0)))) in
  
  (* Pi, variance, skewness terms *)
  let term2 = Tensor.(scalar (float (-0.5)) * scalar (float t_float) * 
                      log (nu_minus_2 * scalar pi) + 
                      scalar (float t_float) * log (scalar (float 2.0) * s / (xi + scalar (float 1.0) / xi))) in
  
  (* Sigma2 term *)
  let term3 = Tensor.(scalar (float (-0.5)) * sum (log sigma2)) in
  
  (* (1 + (sy/sigma+m)^2/(nu-2))^(-(nu+1)/2) term *)
  let scaled_y = Tensor.(s * y / sqrt sigma2 + m) in
  let term4 = Tensor.(scalar (float (-0.5)) * (nu + scalar (float 1.0)) * 
                      sum (log (scalar (float 1.0) + pow scaled_y (scalar (float 2.0)) / 
                      (nu_minus_2 * xi_ind)))) in
  
  Tensor.(term1 + term2 + term3 + term4)

(* Gaussian Variational Approximation *)
module VariationalDist = struct
  type t = {
    mu: Tensor.t;
    l_chol: Tensor.t;
    d: int;
  }
  
  let create ~d ~init_mu ~init_l =
    { mu = init_mu; l_chol = init_l; d }
    
  let sample ~n t =
    let eps = Tensor.randn [n; t.d] in
    Tensor.(t.mu + mm eps t.l_chol)
    
  let log_pdf t theta =
    let d_float = Float.of_int t.d in
    let diff = Tensor.(theta - t.mu) in
    let prec_diff = Tensor.solve t.l_chol diff in
    
    Tensor.(scalar (float (-0.5) *. d_float *. (log (float 2.0 *. Float.pi))) +
            scalar (float (-0.5)) * (sum (pow prec_diff (scalar (float 2.0)))) +
            sum (log (diag t.l_chol)))
    
  let entropy t =
    let d_float = Float.of_int t.d in
    let det_term = Tensor.(sum (log (diag t.l_chol))) in
    Tensor.(scalar (float 0.5) * scalar (float d_float) * 
           (scalar (float 1.0) + scalar (float (log (2.0 *. Float.pi)))) + det_term)
    
  let kl_divergence ~q ~p_log_pdf ~samples =
    let n = Tensor.shape samples |> List.hd in
    let log_q = Tensor.stack (List.init n (fun i -> 
                  log_pdf q (Tensor.get samples i)
                )) ~dim:0 in
    
    let log_p = p_log_pdf samples in
    
    Tensor.((sum log_q - sum log_p) / scalar (float (Float.of_int n)))
end

(* GARCH models *)
module GarchModel = struct
  type innovation_type = Gaussian | StudentT | SkewT
  
  type t = {
    innovation_type: innovation_type;
    y: Tensor.t;
    priors: Tensor.t -> Tensor.t;
    grad_priors: Tensor.t -> Tensor.t;
  }
  
  let create ~innovation_type ~y ~priors ~grad_priors =
    { innovation_type; y; priors; grad_priors }
    
  let compute_sigma2 t params =
    let n_params = Tensor.shape params |> List.hd in
    
    match t.innovation_type with
    | Gaussian ->
        let omega = Tensor.slice params ~dim:0 ~start:0 ~end_:1 ~step:1 in
        let alpha = Tensor.slice params ~dim:0 ~start:1 ~end_:2 ~step:1 in
        let beta = Tensor.slice params ~dim:0 ~start:2 ~end_:3 ~step:1 in
        GarchVolatility.garch_1_1 ~omega ~alpha ~beta ~y:t.y
        
    | StudentT ->
        let omega = Tensor.slice params ~dim:0 ~start:0 ~end_:1 ~step:1 in
        let alpha = Tensor.slice params ~dim:0 ~start:1 ~end_:2 ~step:1 in
        let beta = Tensor.slice params ~dim:0 ~start:2 ~end_:3 ~step:1 in
        GarchVolatility.garch_1_1 ~omega ~alpha ~beta ~y:t.y
        
    | SkewT ->
        let omega = Tensor.slice params ~dim:0 ~start:0 ~end_:1 ~step:1 in
        let alpha = Tensor.slice params ~dim:0 ~start:1 ~end_:2 ~step:1 in
        let beta = Tensor.slice params ~dim:0 ~start:2 ~end_:3 ~step:1 in
        GarchVolatility.garch_1_1 ~omega ~alpha ~beta ~y:t.y
    
  let log_likelihood t params =
    let sigma2 = compute_sigma2 t params in
    
    match t.innovation_type with
    | Gaussian -> 
        LogLikelihood.gaussian ~y:t.y ~sigma2
        
    | StudentT ->
        let nu = Tensor.slice params ~dim:0 ~start:3 ~end_:4 ~step:1 in
        LogLikelihood.student_t ~y:t.y ~sigma2 ~nu
        
    | SkewT ->
        let nu = Tensor.slice params ~dim:0 ~start:3 ~end_:4 ~step:1 in
        let xi = Tensor.slice params ~dim:0 ~start:4 ~end_:5 ~step:1 in
        LogLikelihood.skew_t ~y:t.y ~sigma2 ~nu ~xi
end

(* Stochastic Variational Inference *)
module SVI = struct
  type method_type = ControlVariates | Reparametrization
  
  type optimization_params = {
    learning_rate: float;
    beta1: float;  (* ADAM first moment decay *)
    beta2: float;  (* ADAM second moment decay *)
    max_iter: int;
    tol: float;
    patience: int;
    mc_samples: int;
    window_size: int;
  }
  
  let default_optimization_params = {
    learning_rate = 0.02;
    beta1 = 0.9;
    beta2 = 0.999;
    max_iter = 1000;
    tol = 1e-3;
    patience = 100;
    mc_samples = 10;
    window_size = 25;
  }
  
  (* ELBO computation for control variates approach *)
  let elbo_control_variates ~model ~var_dist ~samples =
    let log_prior = model.priors samples in
    let log_likelihood = 
      Tensor.stack (List.init (Tensor.shape samples |> List.hd) (fun i ->
        GarchModel.log_likelihood model (Tensor.get samples i)
      )) ~dim:0
    in
    let log_q = 
      Tensor.stack (List.init (Tensor.shape samples |> List.hd) (fun i ->
        VariationalDist.log_pdf var_dist (Tensor.get samples i)
      )) ~dim:0
    in
    
    Tensor.(mean (log_prior + log_likelihood - log_q))
  
  (* Stochastic gradient estimation with control variates *)
  let grad_elbo_cv ~model ~var_dist ~samples =
    let h_theta = 
      let log_prior = model.priors samples in
      let log_likelihood = 
        Tensor.stack (List.init (Tensor.shape samples |> List.hd) (fun i ->
          GarchModel.log_likelihood model (Tensor.get samples i)
        )) ~dim:0
      in
      let log_q = 
        Tensor.stack (List.init (Tensor.shape samples |> List.hd) (fun i ->
          VariationalDist.log_pdf var_dist (Tensor.get samples i)
        )) ~dim:0
      in
      
      Tensor.(log_prior + log_likelihood - log_q)
    in
    
    (* Compute grad log q with respect to variational parameters *)
    let grad_log_q_samples = 
      List.init (Tensor.shape samples |> List.hd) (fun i ->
        let sample = Tensor.get samples i in
        let diff = Tensor.(sample - var_dist.mu) in
        
        (* Gradient w.r.t. mu is Sigma^-1 * (theta - mu) *)
        let prec_diff = Tensor.solve var_dist.l_chol diff in
        let grad_mu = prec_diff in
        
        (* Gradient w.r.t. L requires more complex operations *)
        let outer_diff = Tensor.outer diff diff in
        let prec = Tensor.inverse var_dist.l_chol in
        let grad_l = Tensor.(
          scalar (float (-0.5)) * 
          (mm (transpose prec) outer_diff) * prec -
          scalar (float 0.5) * transpose prec
        ) in
        
        (* Combine gradients *)
        Tensor.cat [grad_mu; grad_l] ~dim:0
      ) in
    let grad_log_q = Tensor.stack grad_log_q_samples ~dim:0 in
    
    (* Compute control variates - optimal c minimizes variance *)
    let c = 
      let h_grad = Tensor.(h_theta * grad_log_q) in
      let cov = Tensor.(mean h_grad - mean h_theta * mean grad_log_q) in
      let var_grad = Tensor.(mean (pow grad_log_q (scalar (float 2.0))) - pow (mean grad_log_q) (scalar (float 2.0))) in
      Tensor.(cov / var_grad)
    in
    
    (* Return gradient estimate *)
    Tensor.(mean ((h_theta - c) * grad_log_q))
  
  (* ELBO computation for reparametrization approach *)
  let elbo_reparam ~model ~var_dist ~eps ~mu ~l_chol =
    let samples = Tensor.(mu + mm eps l_chol) in
    
    let log_prior = model.priors samples in
    let log_likelihood = 
      Tensor.stack (List.init (Tensor.shape samples |> List.hd) (fun i ->
        GarchModel.log_likelihood model (Tensor.get samples i)
      )) ~dim:0
    in
    let log_q = 
      Tensor.stack (List.init (Tensor.shape samples |> List.hd) (fun i ->
        VariationalDist.log_pdf var_dist (Tensor.get samples i)
      )) ~dim:0
    in
    
    Tensor.(mean (log_prior + log_likelihood - log_q))
  
  (* Fit variational approximation using control variates *)
  let fit_cv ~model ~init_var_dist ~opt_params =
    let var_dist = ref init_var_dist in
    let iter = ref 0 in
    let patience_counter = ref 0 in
    let best_elbo = ref Float.neg_infinity in
    let elbo_history = ref [] in
    let converged = ref false in
    
    (* Optimization variables *)
    let mu_optim = Optimizer.adam [!var_dist.mu] ~lr:opt_params.learning_rate in
    let l_optim = Optimizer.adam [!var_dist.l_chol] ~lr:opt_params.learning_rate in
    
    while not !converged && !iter < opt_params.max_iter do
      (* Sample from variational distribution *)
      let samples = VariationalDist.sample ~n:opt_params.mc_samples !var_dist in
      
      (* Compute ELBO *)
      let elbo = elbo_control_variates ~model ~var_dist:!var_dist ~samples in
      elbo_history := elbo :: !elbo_history;
      
      if Tensor.to_float0_exn elbo > !best_elbo then (
        best_elbo := Tensor.to_float0_exn elbo;
        patience_counter := 0;
      ) else (
        patience_counter := !patience_counter + 1;
        if !patience_counter >= opt_params.patience then
          converged := true;
      );
      
      (* Compute gradients *)
      let grad = grad_elbo_cv ~model ~var_dist:!var_dist ~samples in
      
      (* Update parameters *)
      Optimizer.backward_step mu_optim;
      Optimizer.backward_step l_optim;
      
      incr iter;
    done;
    
    !var_dist, !elbo_history, !iter
    
  (* Fit variational approximation using reparametrization trick *)
  let fit_reparam ~model ~init_var_dist ~opt_params =
    let var_dist = ref init_var_dist in
    let iter = ref 0 in
    let patience_counter = ref 0 in
    let best_elbo = ref Float.neg_infinity in
    let elbo_history = ref [] in
    let converged = ref false in
    
    (* Optimization variables *)
    let mu_optim = Optimizer.adam [!var_dist.mu] ~lr:opt_params.learning_rate in
    let l_optim = Optimizer.adam [!var_dist.l_chol] ~lr:opt_params.learning_rate in
    
    while not !converged && !iter < opt_params.max_iter do
      (* Generate random samples for reparametrization *)
      let eps = Tensor.randn [opt_params.mc_samples; !var_dist.d] in
      
      (* Compute ELBO with reparametrization *)
      let elbo = elbo_reparam ~model ~var_dist:!var_dist ~eps ~mu:!var_dist.mu ~l_chol:!var_dist.l_chol in
      elbo_history := elbo :: !elbo_history;
      
      if Tensor.to_float0_exn elbo > !best_elbo then (
        best_elbo := Tensor.to_float0_exn elbo;
        patience_counter := 0;
      ) else (
        patience_counter := !patience_counter + 1;
        if !patience_counter >= opt_params.patience then
          converged := true;
      );
      
      (* Compute gradients via reparametrization trick *)
      let mu_grad, l_grad = 
        let open Autograd in
        (* Enable gradient tracking *)
        let mu_tracked = track !var_dist.mu in
        let l_tracked = track !var_dist.l_chol in
        
        (* Compute elbo with tracked parameters *)
        let tracked_elbo = elbo_reparam ~model ~var_dist:!var_dist ~eps ~mu:mu_tracked ~l_chol:l_tracked in
        
        (* Backpropagate to get gradients *)
        backwards tracked_elbo;
        let mu_g = grad mu_tracked |> option_get in
        let l_g = grad l_tracked |> option_get in
        mu_g, l_g
      in
      
      (* Update parameters with gradients *)
      Optimizer.backward_step mu_optim;
      Optimizer.backward_step l_optim;
      
      incr iter;
    done;
    
    !var_dist, !elbo_history, !iter
end

(* Sequential variational inference *)
module SequentialSVI = struct
  (* Updating Variational Bayes (UVB) *)
  let updating_vb ~model ~init_var_dist ~opt_params ~new_data =
    (* Get the previous variational approximation *)
    let prior_var_dist = init_var_dist in
    
    (* Create a new model with only the new data *)
    let new_model = { model with y = new_data } in
    
    (* Create a pseudo-posterior using the prior variational approximation *)
    let pseudo_posterior samples =
      (* Compute log likelihood for new data only *)
      let log_likelihood = 
        Tensor.stack (List.init (Tensor.shape samples |> List.hd) (fun i ->
          GarchModel.log_likelihood new_model (Tensor.get samples i)
        )) ~dim:0
      in
      
      (* Compute log q from prior variational approximation *)
      let log_q_prior = 
        Tensor.stack (List.init (Tensor.shape samples |> List.hd) (fun i ->
          VariationalDist.log_pdf prior_var_dist (Tensor.get samples i)
        )) ~dim:0
      in
      
      Tensor.(log_likelihood + log_q_prior)
    in
    
    (* Create a new gradient function based on the pseudo-posterior *)
    let grad_pseudo_posterior samples =
      let n = Tensor.shape samples |> List.hd in
      Tensor.zeros [n; init_var_dist.d]
    in
    
    (* Create a new model with the pseudo-posterior *)
    let uvb_model = { new_model with 
                      priors = pseudo_posterior; 
                      grad_priors = grad_pseudo_posterior } in
    
    (* Run variational inference with the pseudo-posterior model *)
    SVI.fit_reparam ~model:uvb_model ~init_var_dist ~opt_params
  
  (* Sequential Stochastic Variational Bayes (Seq-SVB) *)
  let sequential_svb ~model ~init_var_dist ~opt_params ~new_data =
    (* Create a new model with all data *)
    let all_data = Tensor.cat [model.y; new_data] ~dim:0 in
    let updated_model = { model with y = all_data } in
    
    (* Run variational inference with all data, using previous variational approximation as starting point *)
    SVI.fit_reparam ~model:updated_model ~init_var_dist ~opt_params
    
  (* Function for sequential updating with chunks of data *)
  let sequential_update ~model ~init_var_dist ~opt_params ~data_chunks ~method_type =
    let var_dist = ref init_var_dist in
    let elbo_history = ref [] in
    let timing_results = ref [] in
    
    (* Process each data chunk sequentially *)
    List.iteri (fun i chunk ->
      Printf.printf "Processing chunk %d of %d\n" (i+1) (List.length data_chunks);
      let start_time = Unix.gettimeofday () in
      
      let updated_var_dist, chunk_elbo_history, iterations = 
        match method_type with
        | `UVB -> updating_vb ~model ~init_var_dist:!var_dist ~opt_params ~new_data:chunk
        | `Seq_SVB -> sequential_svb ~model ~init_var_dist:!var_dist ~opt_params ~new_data:chunk
      in
      
      let end_time = Unix.gettimeofday () in
      let processing_time = end_time -. start_time in
      
      var_dist := updated_var_dist;
      elbo_history := !elbo_history @ chunk_elbo_history;
      timing_results := processing_time :: !timing_results;
      
      Printf.printf "Chunk %d processed in %.2f seconds, %d iterations\n" 
        (i+1) processing_time iterations;
    ) data_chunks;
    
    !var_dist, !elbo_history, !timing_results
end