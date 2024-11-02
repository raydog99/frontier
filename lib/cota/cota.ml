open Torch
open Types

type config = {
  lambda: float;
  gamma: float;
  epsilon: float;
  max_iter: int;
  lr: float;
  tol: float;
}

let chain_objective plans chain cost config =
  (* OT cost term *)
  let ot_cost = List.fold_left (fun acc plan ->
    Tensor.(add acc (mul_scalar (sum (mul plan.plan cost)) config.lambda))
  ) (Tensor.zeros []) plans in
  
  (* Do-calculus constraints *)
  let dc_cost = List.fold_left2 (fun acc plan1 plan2 i1 i2 ->
    let dist = CausalConstraints.do_calculus_distance plan1 plan2 i1 i2 in
    Tensor.(add acc (mul_scalar dist config.gamma))
  ) (Tensor.zeros []) (List.tl plans) plans (List.tl chain) chain in
  
  (* Entropy regularization *)
  let ent_cost = List.fold_left (fun acc plan ->
    Tensor.(add acc (mul_scalar (OptimalTransport.entropy plan.plan) config.epsilon))
  ) (Tensor.zeros []) plans in
  
  Tensor.(add (add ot_cost dc_cost) ent_cost)

let optimize_chain chain base_samples abstract_samples omega config =
  let cost = OptimalTransport.cost_matrix base_samples abstract_samples chain omega in
  
  (* Initialize plans *)
  let initial_plans = List.map (fun intervention ->
    let n_source = Tensor.(shape base_samples |> List.hd) in
    let n_target = Tensor.(shape abstract_samples |> List.hd) in
    let source_dist = Tensor.(ones [n_source] / float n_source) in
    let target_dist = Tensor.(ones [n_target] / float n_target) in
    let plan = OptimalTransport.sinkhorn cost source_dist target_dist config.epsilon in
    {plan; source_dist; target_dist}
  ) chain in
  
  let rec iterate plans iter prev_obj =
    if iter >= config.max_iter then plans
    else
      (* Compute objective and gradients *)
      let obj = chain_objective plans chain cost config in
      let obj_val = Tensor.item obj in
      
      (* Check convergence *)
      if abs_float (obj_val -. prev_obj) < config.tol then plans
      else
        (* Update each plan *)
        let updated_plans = List.map2 (fun plan intervention ->
          let grad = Tensor.grad obj [plan.plan] in
          let updated_plan = OptimalTransport.gradient_step plan.plan (List.hd grad) config.lr in
          {plan with plan = updated_plan}
        ) plans chain in
        
        iterate updated_plans (iter + 1) obj_val
  in
  
  iterate initial_plans 0 Float.infinity

let create_abstraction_map chains_plans omega =
  (* Convert each plan to a stochastic map *)
  let chain_maps = List.map (fun plans ->
    List.map (fun plan -> StochasticMap.create_from_plan plan.plan) plans
  ) chains_plans in
  
  (* Average maps across chains *)
  let final_map = StochasticMap.average_maps (List.flatten chain_maps) in
  
  {tau = final_map; omega}

let optimize base_scm abstract_scm interventions config =
  let maximal_chains = Chain.compute_maximal_chains interventions in
  
  (* Generate samples *)
  let base_samples = Scm.sample base_scm 1000 in
  let abstract_samples = Scm.sample abstract_scm 1000 in
  
  (* Create omega map based on variable relationships *)
  let var_mapping = List.mapi (fun i v -> (i, i)) (Array.to_list base_scm.variables) in
  let omega = Intervention.create_omega_map base_scm abstract_scm var_mapping in
  
  (* Optimize each chain *)
  let chain_plans = List.map (fun chain ->
    optimize_chain chain base_samples abstract_samples omega config
  ) maximal_chains in
  
  create_abstraction_map chain_plans omega