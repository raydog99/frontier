open Torch
open Type

let acceptance_prob x y theta_new theta_old sigma config =
  let log_likelihood pred =
    let n = float_of_int (size x 0) in
    let residuals = sub y pred in
    let ss = dot residuals residuals in
    -0.5 *. (n *. log (2. *. Float.pi *. sigma) +. float_of_elt ss /. sigma)
  in
  
  let new_ll = log_likelihood (matmul x theta_new) in
  let old_ll = log_likelihood (matmul x theta_old) in
  exp (min 0. (new_ll -. old_ll))

let adapt_step_size config acc_rate =
  let delta = acc_rate -. config.target_acceptance in
  config.step_size := !(config.step_size) *. exp (0.999 *. delta)

let run x y config =
  let samples = ref [] in
  let acc_count = ref 0 in
  
  let init_theta = Posterior.sample_posterior x y config.base_config 1.0 in
  let init_theta_star = Projection.sparse_projection x init_theta config.base_config.lambda in
  let sigma = ref (SigmaEstimation.estimate_sigma x y init_theta_star) in
  
  let current_theta = ref init_theta in
  
  for i = 1 to config.base_config.iterations do
    (* Propose new theta *)
    let noise = scalar_mul !(config.step_size) (randn_like !current_theta) in
    let proposed_theta = add !current_theta noise in
    
    (* Accept/reject *)
    let acc_prob = acceptance_prob x y proposed_theta !current_theta !sigma config.base_config in
    if Random.float 1.0 < acc_prob then begin
      current_theta := proposed_theta;
      acc_count := !acc_count + 1
    end;
    
    (* Project and store *)
    let theta_star = Projection.sparse_projection x !current_theta config.base_config.lambda in
    let theta_debiased = DebiasedProjection.debiased_projection x !current_theta theta_star config.base_config in
    
    if i > config.base_config.burn_in then
      samples := {theta = !current_theta; theta_star; theta_debiased; sigma = !sigma} :: !samples;
    
    (* Adapt step size during adaptation period *)
    if i <= config.adaptation_period then
      adapt_step_size config (float_of_int !acc_count /. float_of_int i);
      
    sigma := SigmaEstimation.sample_sigma x y theta_star (size x 0)
  done;
  
  !samples