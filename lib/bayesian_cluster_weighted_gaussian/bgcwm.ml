open Torch

type params = {
  n: int;
  p: int;
  k: int;
  pi: Tensor.t;
  alpha: Tensor.t;
  beta: Tensor.t;
  sigma_sq: Tensor.t;
  mu: Tensor.t;
  sigma: Tensor.t;
}

type prior_params = {
  dir_alpha: float;
  bnb_a: float;
  bnb_a_pi: float;
  bnb_b_pi: float;
  f_nu_l: float;
  f_nu_r: float;
  alpha_var: float;
  sigma_shape: float;
  sigma_rate: float;
  mu_mean: Tensor.t;
  psi_shape: float;
  psi_rate: float;
}

type mcmc_state = {
  params: params;
  priors: prior_params;
  z: Tensor.t;
  log_likelihood: float;
  iteration: int;
}

module Distributions = struct
  let sample_dirichlet alpha k =
    let gammas = List.init k (fun _ -> 
      Random.gamma (alpha /. float_of_int k) 1.0) in
    let sum = List.fold_left (+.) 0. gammas in
    List.map (fun x -> x /. sum) gammas |>
    Tensor.of_float1

  let sample_inverse_gamma shape rate size =
    let samples = List.init (List.fold_left ( * ) 1 size) (fun _ ->
      1.0 /. Random.gamma shape (1.0 /. rate)) in
    Tensor.of_float_list samples |> 
    Tensor.reshape (Array.of_list size)

  let sample_wishart df scale =
    let p = Tensor.size scale ~dim:0 in
    let l = Tensor.cholesky scale in
    let a = Tensor.zeros [p; p] in
    
    for i = 0 to p - 1 do
      let chi_df = df -. float_of_int i in
      let x = Random.gamma (chi_df /. 2.0) 2.0 in
      Tensor.set a [i;i] (sqrt x)
    done;
    
    for i = 0 to p - 2 do
      for j = i + 1 to p - 1 do
        Tensor.set a [i;j] (Random.gaussian())
      done
    done;
    
    let la = Tensor.mm l a in
    Tensor.mm la (Tensor.transpose la ~dim0:0 ~dim1:1)

  let sample_beta_negative_binomial a a_pi b_pi =
    let beta = Random.beta a_pi b_pi in
    let rec sample_nbinom () =
      let u = Random.float 1.0 in
      let p = beta in
      let log_q = log (1. -. p) in
      let x = ref 0 in
      let sum = ref 0. in
      while !sum > log u do
        incr x;
        sum := !sum +. log (a +. float_of_int (!x - 1)) -.
               log (float_of_int !x) +. log p
      done;
      !x
    in sample_nbinom ()

  let sample_half_cauchy scale size =
    let samples = List.init (List.fold_left ( * ) 1 size) (fun _ ->
      let u = Random.float 1.0 in
      scale *. tan (Float.pi *. (u -. 0.5))
    ) in
    Tensor.of_float_list samples |>
    Tensor.reshape (Array.of_list size)
end

module Numerical = struct
  exception NumericalError of string

  let epsilon = 1e-10

  let safe_cholesky mat =
    let n = Tensor.size mat ~dim:0 in
    let jitter = ref epsilon in
    let rec attempt () =
      try
        let stabilized = Tensor.add mat 
          (Tensor.eye n |> Tensor.mul_scalar !jitter) in
        Tensor.cholesky stabilized
      with _ ->
        if !jitter > 1e-3 then
          raise (NumericalError "Matrix not positive definite")
        else begin
          jitter := !jitter *. 10.0;
          attempt ()
        end
    in attempt ()

  let safe_inverse mat =
    let n = Tensor.size mat ~dim:0 in
    let svd = Tensor.svd mat in
    let min_sv = Tensor.min svd ~dim:0 ~keepdim:false |> Tensor.to_float0 in
    if min_sv < epsilon then
      Tensor.add mat (Tensor.eye n |> Tensor.mul_scalar epsilon) |>
      Tensor.inverse
    else
      Tensor.inverse mat

  let safe_log_det mat =
    let chol = safe_cholesky mat in
    let diag = Tensor.diagonal chol ~dim1:0 ~dim2:1 in
    Tensor.mul_scalar 2.0 (Tensor.sum (Tensor.log (Tensor.abs diag))) |>
    Tensor.to_float0

  let log_sum_exp x =
    let max_x = Tensor.max x ~dim:0 ~keepdim:true in
    let shifted = Tensor.sub x max_x in
    Tensor.add max_x (Tensor.log (Tensor.sum (Tensor.exp shifted) ~dim:0))
end

module VariableSelection = struct
  type credible_region = {
    lower: Tensor.t;
    upper: Tensor.t;
  }

  let compute_credible_regions samples alpha =
    let betas = List.map (fun s -> s.params.beta) samples in
    let beta_tensor = Tensor.stack betas ~dim:0 in
    
    let lower = Tensor.quantile beta_tensor (alpha /. 2.) ~dim:0 ~keepdim:false in
    let upper = Tensor.quantile beta_tensor (1. -. alpha /. 2.) ~dim:0 ~keepdim:false in
    
    { lower; upper }

  let select_variables regions =
    let k = Tensor.size regions.lower ~dim:0 in
    let p = Tensor.size regions.lower ~dim:1 in
    let significant = Tensor.zeros [p] ~kind:Bool in
    
    for j = 0 to p - 1 do
      let is_sig = ref false in
      for c = 0 to k - 1 do
        let lower = Tensor.get regions.lower [c;j] |> Tensor.to_float0 in
        let upper = Tensor.get regions.upper [c;j] |> Tensor.to_float0 in
        if lower *. upper > 0.0 then is_sig := true
      done;
      if !is_sig then
        Tensor.set significant [j] (Tensor.of_int0 1)
    done;
    significant
end

module BGCWM = struct
  type t = {
    params: params;
    priors: prior_params
  }

  let update_component_parameters state ~x ~y ~k =
    let mask = Tensor.get state.z k in
    let x_k = Tensor.masked_select x mask |> 
              Tensor.reshape [-1; state.params.p] in
    let y_k = Tensor.masked_select y mask |> 
              Tensor.reshape [-1; 1] in
    let n_k = Tensor.size x_k ~dim:0 in
    
    if n_k > 0 then begin
      (* Update regression coefficients *)
      let sigma_k = Tensor.get state.params.sigma_sq k in
      let precision = Tensor.add
        (Tensor.mm (Tensor.transpose x_k ~dim0:0 ~dim1:1)
           (Tensor.div x_k sigma_k))
        (Tensor.eye state.params.p |>
         Tensor.mul_scalar (1. /. state.priors.alpha_var)) in
      let covariance = Numerical.safe_inverse precision in
      let mean = Tensor.mm covariance
        (Tensor.mm (Tensor.transpose x_k ~dim0:0 ~dim1:1)
           (Tensor.div y_k sigma_k)) in
      let beta_k = Tensor.add mean
        (Tensor.mm (Numerical.safe_cholesky covariance)
           (Tensor.randn [state.params.p; 1])) in
           
      (* Update means *)
      let x_mean = Tensor.mean x_k ~dim:0 ~keepdim:true in
      let mu_cov = Tensor.div state.params.sigma sigma_k in
      let mu_k = MVN.sample 1 ~mu:x_mean ~sigma:mu_cov in
      
      (* Update covariance *)
      let centered = Tensor.sub x_k mu_k in
      let s = Tensor.mm (Tensor.transpose centered ~dim0:0 ~dim1:1)
                       centered in
      let psi_k = Tensor.get state.params.psi k in
      let sigma_k = Distributions.sample_wishart
        (float_of_int n_k)
        (Tensor.div s (float_of_int n_k)) in
        
      Some (beta_k, mu_k, sigma_k)
    end else
      None

  let update_mixing_proportions state =
    let cluster_counts = Tensor.sum state.z ~dim:0 ~keepdim:false in
    let alpha = Tensor.add_scalar cluster_counts state.priors.dir_alpha in
    Distributions.sample_dirichlet
      (Tensor.to_float1 alpha |> Array.to_list)
      state.params.k

  let compute_component_likelihood ~x ~y ~k params =
    let beta_k = Tensor.get params.beta k in
    let sigma_k = Tensor.get params.sigma_sq k in
    let mu_k = Tensor.get params.mu k in
    let cov_k = Tensor.get params.sigma k in
    
    (* Regression likelihood *)
    let mean = Tensor.add
      (Tensor.get params.alpha k)
      (Tensor.mm x (Tensor.reshape beta_k [-1; 1])) in
    let log_lik_y = Tensor.normal_log_prob y ~mean ~std:(sqrt sigma_k) in
    
    (* Covariate likelihood *)
    let centered = Tensor.sub x mu_k in
    let log_lik_x = Tensor.sum
      (Tensor.mm (Tensor.mm centered (Numerical.safe_inverse cov_k))
         (Tensor.transpose centered ~dim0:0 ~dim1:1))
      ~dim:1 ~keepdim:true in
    
    Tensor.add log_lik_y log_lik_x

  let sample_indicators state ~x ~y =
    let n = Tensor.size x ~dim:0 in
    let log_probs = Tensor.zeros [n; state.params.k] in
    
    for k = 0 to state.params.k - 1 do
      let comp_ll = compute_component_likelihood ~x ~y ~k state.params in
      let log_pi_k = Tensor.log (Tensor.get state.params.pi k) in
      Tensor.set_slice_2d log_probs [[];[k]]
        (Tensor.add comp_ll log_pi_k)
    done;
    
    let probs = Tensor.softmax log_probs ~dim:1 in
    Tensor.multinomial probs 1 ~replacement:true

  let gibbs_step state ~x ~y =
    (* Sample indicators *)
    let z = sample_indicators state ~x ~y in
    
    (* Update mixing proportions *)
    let pi = update_mixing_proportions {state with z} in
    
    (* Update component parameters *)
    let new_beta = Tensor.zeros_like state.params.beta in
    let new_mu = Tensor.zeros_like state.params.mu in
    let new_sigma = Tensor.zeros_like state.params.sigma in
    
    for k = 0 to state.params.k - 1 do
      match update_component_parameters {state with z} ~x ~y ~k with
      | Some (beta_k, mu_k, sigma_k) ->
          Tensor.set new_beta k beta_k;
          Tensor.set new_mu k mu_k;
          Tensor.set new_sigma k sigma_k
      | None -> ()
    done;
    
    let new_params = {
      state.params with
      pi;
      beta = new_beta;
      mu = new_mu;
      sigma = new_sigma;
    } in
    
    let log_lik = compute_log_likelihood new_params ~x ~y in
    
    { state with
      params = new_params;
      z;
      log_likelihood = log_lik;
      iteration = state.iteration + 1
    }

  let create ~n ~p ~k =
    let priors = {
      dir_alpha = 1.0;
      bnb_a = 1.0;
      bnb_a_pi = 4.0;
      bnb_b_pi = 3.0;
      f_nu_l = 6.0;
      f_nu_r = 3.0;
      alpha_var = 1e3;
      sigma_shape = 1e-3;
      sigma_rate = 1e-3;
      mu_mean = Tensor.zeros [p];
      psi_shape = 1.0;
      psi_rate = 0.01;
    } in
    
    let params = {
      n; p; k;
      pi = Distributions.sample_dirichlet priors.dir_alpha k;
      alpha = Tensor.randn [k];
      beta = Tensor.randn [k; p];
      sigma_sq = Tensor.ones [k];
      mu = Tensor.randn [k; p];
      sigma = Tensor.stack (List.init k (fun _ -> Tensor.eye p)) ~dim:0;
    } in
    
    { params; priors }

  let init_mcmc ~x ~y =
    let n = Tensor.size x ~dim:0 in
    let p = Tensor.size x ~dim:1 in
    let model = create ~n ~p ~k:2 in
    
    let z = Tensor.randint 0 model.params.k [n] ~dtype:Int64 in
    let log_lik = compute_log_likelihood model.params ~x ~y in
    
    {
      params = model.params;
      priors = model.priors;
      z;
      log_likelihood = log_lik;
      iteration = 0
    }

  let run_mcmc ~x ~y ~init_state ~n_iter =
    let states = ref [init_state] in
    let current_state = ref init_state in
    
    for i = 1 to n_iter do
      current_state := gibbs_step !current_state ~x ~y;
      if i mod 10 = 0 then
        states := !current_state :: !states
    done;
    
    List.rev !states

  let run_analysis ~x ~y ~k_min ~k_max ~n_iter ~alpha =
    let init_state = init_mcmc ~x ~y in
    let samples = run_mcmc ~x ~y ~init_state ~n_iter in
    
    let regions = VariableSelection.compute_credible_regions samples alpha in
    let significant_vars = VariableSelection.select_variables regions in
    
    samples, significant_vars, regions

  let summarize_results samples significant_vars =
    let n_sig = Tensor.sum significant_vars |> Tensor.to_int0 in
    Printf.printf "Number of significant variables: %d\n" n_sig;
    
    let k_counts = Hashtbl.create 10 in
    List.iter (fun s ->
      let k = s.params.k in
      Hashtbl.replace k_counts k
        (1 + try Hashtbl.find k_counts k with Not_found -> 0)
    ) samples;
    
    Printf.printf "Distribution of K:\n";
    Hashtbl.iter (fun k count ->
      Printf.printf "K=%d: %d samples\n" k count
    ) k_counts

  let check_convergence samples =
    let n_samples = List.length samples in
    let mid_point = n_samples / 2 in
    let first_half = List.filteri (fun i _ -> i < mid_point) samples in
    let second_half = List.filteri (fun i _ -> i >= mid_point) samples in
    
    let compute_r_hat param_getter =
      let params1 = List.map param_getter first_half in
      let params2 = List.map param_getter second_half in
      let mean1 = Tensor.mean (Tensor.stack params1 ~dim:0) ~dim:0 in
      let mean2 = Tensor.mean (Tensor.stack params2 ~dim:0) ~dim:0 in
      Tensor.mean (Tensor.abs (Tensor.sub mean1 mean2)) |> Tensor.to_float0
    in
    
    compute_r_hat (fun s -> s.params.beta),
    compute_r_hat (fun s -> s.params.mu)
end

module BatchProcessor = struct
  type batch = {
    x: Tensor.t;
    y: Tensor.t;
    start_idx: int;
    size: int;
  }

  let create_batches ~x ~y ~batch_size =
    let n = Tensor.size x ~dim:0 in
    let n_batches = (n + batch_size - 1) / batch_size in
    Array.init n_batches (fun i ->
      let start = i * batch_size in
      let size = min batch_size (n - start) in
      {
        x = Tensor.narrow x ~dim:0 ~start ~length:size;
        y = Tensor.narrow y ~dim:0 ~start ~length:size;
        start_idx = start;
        size;
      }
    )

  let process_batch model batch =
    try
      let log_lik = BGCWM.compute_log_likelihood model.params 
                      ~x:batch.x ~y:batch.y in
      Ok log_lik
    with
    | Numerical.NumericalError msg -> Error ("Numerical error in batch: " ^ msg)
    | e -> Error ("Unexpected error in batch: " ^ Printexc.to_string e)
end