open Torch
open Types
open Kalman_filter

let log_likelihood model observations maturities =
  let kalman_states = kalman_filter model (Tensor.zeros [model.state_dim]) observations maturities in
  let log_likelihood_sum = ref 0. in
  let n = List.length kalman_states in
  
  for t = 0 to n - 1 do
    let state = List.nth kalman_states t in
    let h = Dns_model.factor_loadings model maturities in
    let innovation = Tensor.(observations.[(t, 0)] - (h * state.mean)) in
    let s = Tensor.(matmul (matmul h state.covariance) (transpose h ~dim0:0 ~dim1:1) + model.params.sigma_epsilon) in
    let log_det_s = Tensor.logdet s |> Tensor.to_float0_exn in
    let weighted_innovation = Tensor.(matmul (transpose innovation ~dim0:0 ~dim1:1) (matmul (inverse s) innovation)) in
    log_likelihood_sum := !log_likelihood_sum -. 0.5 *. (log_det_s +. (Tensor.to_float0_exn weighted_innovation) +. 
                                                         (float model.obs_dim) *. (log 2. *. Float.pi))
  done;
  !log_likelihood_sum

let optimize_mle init_model observations maturities learning_rate num_iterations =
  let params = init_model.params in
  let optimizer = Optimizer.adam [params.beta_0; params.beta_1; params.sigma_epsilon; params.sigma_eta] ~lr:learning_rate in
  
  let rec optimize iter best_model best_ll =
    if iter = num_iterations then
      best_model
    else
      let loss = Tensor.negf (log_likelihood init_model observations maturities) in
      Optimizer.zero_grad optimizer;
      Tensor.backward loss;
      Optimizer.step optimizer;
      
      let current_ll = log_likelihood init_model observations maturities in
      let new_best_model, new_best_ll = 
        if current_ll > best_ll then (init_model, current_ll) else (best_model, best_ll)
      in
      
      optimize (iter + 1) new_best_model new_best_ll
  in
  
  optimize 0 init_model (log_likelihood init_model observations maturities)

let estimate_parameters init_model observations maturities learning_rate num_iterations =
  optimize_mle init_model observations maturities learning_rate num_iterations