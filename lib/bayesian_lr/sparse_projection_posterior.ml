open Torch
open Type

let fit x y config n_chains =
  let chain_states = Array.init n_chains (fun _ ->
    let samples = ref [] in
    
    let init_theta = Posterior.sample_posterior x y config 1.0 in
    let init_theta_star = Projection.sparse_projection x init_theta config.lambda in
    let sigma = ref (SigmaEstimation.estimate_sigma x y init_theta_star) in
    
    for i = 1 to config.iterations do
      let theta = Posterior.sample_posterior x y config !sigma in
      let theta_star = Projection.sparse_projection x theta config.lambda in
      let theta_debiased = DebiasedProjection.debiased_projection x theta theta_star config in
      
      if i > config.burn_in then
        samples := {theta; theta_star; theta_debiased; sigma = !sigma} :: !samples;
        
      sigma := SigmaEstimation.sample_sigma x y theta_star (size x 0)
    done;
    Array.of_list !samples
  ) in
  
  (* Assess convergence *)
  let all_samples = Array.fold_left (fun acc chain ->
    Array.to_list chain @ acc
  ) [] chain_states in
  
  let diagnostics = Convergence.assess_convergence all_samples chain_states in
  
  chain_states, diagnostics

let fit_cv x y base_config k =
  (* Find optimal lambda *)
  let cv_results = CrossValidation.find_optimal_lambda x y base_config k in
  let optimal_lambda = (List.hd cv_results).lambda in
  
  (* Fit final model with optimal lambda *)
  let final_config = {base_config with lambda = optimal_lambda} in
  let chains, diagnostics = fit x y final_config 1 in
  
  chains, diagnostics, cv_results

let fit_adaptive x y base_config =
  let adaptive_config = {
    base_config;
    adaptation_period = 5000;
    target_acceptance = 0.234;  (* Optimal rate for multivariate case *)
    step_size = ref 0.1;
  } in
  AdaptiveMCMC.run x y adaptive_config

let fit_parallel x y config n_chains =
  let chain_states = Parallel.run_parallel_chains x y config n_chains in
  
  (* Assess convergence *)
  let all_samples = Array.fold_left (fun acc state ->
    state.samples @ acc
  ) [] chain_states in
  
  let diagnostics = Convergence.assess_convergence all_samples 
    (Array.map (fun state -> Array.of_list state.samples) chain_states) in
  
  chain_states, diagnostics

let test_parameters samples parameters null_values alpha =
  List.map2 (fun idx null_value ->
    HypothesisTest.test_parameter samples idx null_value alpha
  ) parameters null_values

let test_joint_hypothesis samples indices null_values alpha =
  HypothesisTest.test_joint samples indices null_values alpha

let compare_models samples x y =
  let dic_stats = ModelComparison.compute_dic samples x y in
  let waic_stats = ModelComparison.compute_waic samples x y in
  {
    dic = dic_stats.dic;
    waic = waic_stats.waic;
    lpd = (dic_stats.lpd +. waic_stats.lpd) /. 2.;
    p_eff = (dic_stats.p_eff +. waic_stats.p_eff) /. 2.;
  }