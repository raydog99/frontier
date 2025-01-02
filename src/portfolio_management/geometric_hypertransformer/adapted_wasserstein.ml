open Torch

type transport_plan = {
  coupling: Tensor.t;
  marginals: Tensor.t * Tensor.t;
  cost: float;
  bicausality_violation: float;
}

type optimal_transport_params = {
  epsilon: float;
  max_iterations: int;
  tolerance: float;
}

let compute_bicausal_transport ~source ~target ~filtration ~params =
  let n = Tensor.size source 0 in
  let m = Tensor.size target 0 in
  
  let alpha = Tensor.zeros [n] in
  let beta = Tensor.zeros [m] in
  let coupling = ref (Tensor.zeros [n; m]) in
  
  let cost = Tensor.cdist source target ~p:2. in
  let masked_cost = 
    Tensor.where filtration.causality_mask cost 
      (Tensor.full_like cost Float.infinity) in
  
  let kernel = Tensor.exp (Tensor.div_scalar 
    (Tensor.neg masked_cost) params.epsilon) in
  
  let rec iterate alpha beta iter prev_cost =
    if iter >= params.max_iterations then alpha, beta
    else
      let alpha' = 
        Tensor.log (Tensor.div source (
          Tensor.mm kernel (Tensor.exp beta))) in
      let beta' = 
        Tensor.log (Tensor.div target (
          Tensor.mm (Tensor.transpose kernel) 
            (Tensor.exp alpha))) in
      
      let curr_cost = 
        Tensor.sum (Tensor.mul kernel 
          (Tensor.exp (Tensor.add 
            (Tensor.reshape alpha [n; 1])
            (Tensor.reshape beta [1; m])))) 
        |> Tensor.float_value in
      
      if abs_float (curr_cost -. prev_cost) < params.tolerance
      then alpha', beta'
      else iterate alpha' beta' (iter + 1) curr_cost
  in
  
  let alpha_opt, beta_opt = iterate alpha beta 0 Float.infinity in
  
  coupling := Tensor.mul kernel (
    Tensor.exp (Tensor.add 
      (Tensor.reshape alpha_opt [n; 1])
      (Tensor.reshape beta_opt [1; m])));
  
  let violation = 
    Tensor.sum (Tensor.relu (
      Tensor.sub !coupling 
        (Tensor.ones_like !coupling))) 
    |> Tensor.float_value in
  
  {
    coupling = !coupling;
    marginals = (source, target);
    cost = Tensor.sum (Tensor.mul !coupling masked_cost) 
           |> Tensor.float_value;
    bicausality_violation = violation;
  }

let adapted_wasserstein_distance ~p plan =
  let temporal_weight t = 
    exp (-. float_of_int t) in
  
  let cost = ref 0. in
  let time_steps = Tensor.size plan.coupling 1 in
  
  for t = 0 to time_steps - 1 do
    let coupling_t = 
      Tensor.select plan.coupling ~dim:1 ~index:t in
    let source_t = 
      Tensor.select (fst plan.marginals) ~dim:1 ~index:t in
    let target_t = 
      Tensor.select (snd plan.marginals) ~dim:1 ~index:t in
    
    let local_cost = 
      Tensor.sum (Tensor.mul coupling_t 
        (Tensor.cdist source_t target_t ~p:(float_of_int p))) in
    
    cost := !cost +. temporal_weight t *. 
            (Tensor.float_value local_cost)
  done;
  
  !cost ** (1. /. float_of_int p) +. 
  plan.bicausality_violation

let verify_bicausality plan filtration =
  let past_coupling = 
    Tensor.narrow plan.coupling 
      ~dim:0 ~start:0 
      ~length:(Tensor.size filtration.history 0) in
  
  Tensor.sum (Tensor.relu (
    Tensor.sub past_coupling 
      (Tensor.ones_like past_coupling))) 
  |> Tensor.float_value