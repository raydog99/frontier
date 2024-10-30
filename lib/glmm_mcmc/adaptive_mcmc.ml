open Torch
open Types

type adaptive_state = {
  mcmc_state: mcmc_state;
  covariance: Tensor.t;
  mean: Tensor.t;
  samples: model_params list;
  adaptation_count: int;
}

let update_moments state =
  let n = float_of_int state.adaptation_count in
  let params_tensor = Tensor.cat [
    state.mcmc_state.params.beta;
    state.mcmc_state.params.u;
    state.mcmc_state.params.lambda;
  ] 0 in
  
  (* Update mean *)
  let new_mean = Tensor.(
    add
      (mul state.mean (float_tensor [n /. (n +. 1.0)]))
      (mul params_tensor (float_tensor [1.0 /. (n +. 1.0)]))
  ) in
  
  (* Update covariance *)
  let centered = Tensor.(sub params_tensor new_mean) in
  let new_cov = Tensor.(
    add
      (mul state.covariance (float_tensor [n /. (n +. 1.0)]))
      (mm (unsqueeze centered 1) (unsqueeze centered 0))
  ) in
  
  {state with 
    mean = new_mean;
    covariance = new_cov;
    adaptation_count = state.adaptation_count + 1}

let adapt_step data state =
  (* Update adaptation parameters *)
  let updated_state = update_moments state in
  
  (* Compute new step size *)
  let acceptance_rate = 
    float_of_int state.mcmc_state.accepted /. 
    float_of_int state.mcmc_state.total in
  let target_rate = 0.234 in  (* Optimal acceptance rate for high-dim problems *)
  let log_epsilon = 
    log state.mcmc_state.epsilon +.
    1.0 /. sqrt (float_of_int state.adaptation_count) *. 
    (acceptance_rate -. target_rate) in
  
  (* Use adapted covariance in MALA step *)
  let scaled_cov = Tensor.(
    mul
      (float_tensor [2.38 *. 2.38 /. float_of_int (size state.covariance 0)])
      state.covariance
  ) in
  
  let new_mcmc_state = MALA.manifold_step data 
    {state.mcmc_state with epsilon = exp log_epsilon}
    scaled_cov in
  
  {updated_state with mcmc_state = new_mcmc_state}

let run data init_state n_adapt n_samples =
  (* Adaptation phase *)
  let rec adapt state i =
    if i >= n_adapt then state
    else
      let new_state = adapt_step data state in
      adapt new_state (i + 1)
  in
  
  (* Sampling phase with fixed parameters *)
  let final_adapted = adapt init_state 0 in
  let rec sample state samples i =
    if i >= n_samples then List.rev samples
    else
      let new_state = MALA.step data state.mcmc_state state.mcmc_state.epsilon in
      sample {state with mcmc_state = new_state} 
        (new_state.params :: samples) (i + 1)
  in
  
  sample final_adapted [] 0