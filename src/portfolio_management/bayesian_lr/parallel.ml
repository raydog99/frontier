open Torch
open Type

type chain_state = {
  samples: posterior_sample list;
  rng_state: Random.State.t;
  id: int;
}

let init_chains n_chains =
  Array.init n_chains (fun id ->
    let rng_state = Random.State.make [|id|] in
    {samples = []; rng_state; id}
  )

let run_chain x y config state =
  Random.set_state state.rng_state;
  
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
  
  {state with 
    samples = !samples;
    rng_state = Random.get_state ()
  }

let run_parallel_chains x y config n_chains =
  let chains = init_chains n_chains in
  
  (* Create domains for parallel execution *)
  let domains = Array.init n_chains (fun _ -> Domain.spawn (fun () ->
    run_chain x y config (Array.get chains 0)
  )) in
  
  (* Collect results *)
  Array.map Domain.join domains