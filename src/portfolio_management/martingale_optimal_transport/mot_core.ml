open Torch

module DiscreteMeasure : Measure = struct
  type t = {
    support: Tensor.t;
    density: Tensor.t;
  }

  let create ?support density =
    let sup = match support with
      | Some s -> s
      | None -> 
          let n = (Tensor.shape density).(0) in
          Tensor.linspace 0.0 1.0 n
    in
    if Tensor.float_value (Tensor.sum density) <> 1.0 then
      failwith "Density must sum to 1";
    {support = sup; density}

  let support m = m.support
  let density m = Some m.density

  let sample m n =
    let indices = Tensor.multinomial m.density ~num_samples:n ~replacement:true in
    Tensor.index_select m.support ~dim:0 ~index:indices

  let expectation m f =
    let values = f m.support in
    Tensor.dot values m.density

  let wasserstein_distance m1 m2 =
    let cost_matrix = Tensor.cdist m1.support m2.support ~p:2.0 in
    let n1 = (Tensor.shape m1.support).(0) in
    let n2 = (Tensor.shape m2.support).(0) in
    
    (* Setup LP for optimal transport *)
    let vars = Tensor.zeros [|n1; n2|] in
    let obj = Tensor.reshape cost_matrix [|-1|] in
    
    (* Equality constraints for marginals *)
    let a1 = Tensor.zeros [|n1; n1 * n2|] in
    let a2 = Tensor.zeros [|n2; n1 * n2|] in
    
    for i = 0 to n1-1 do
      for j = 0 to n2-1 do
        Tensor.copy_ 
          (Tensor.narrow a1 ~dim:0 ~start:i ~length:1)
          (Tensor.ones [|1|]);
        Tensor.copy_
          (Tensor.narrow a2 ~dim:0 ~start:j ~length:1)
          (Tensor.ones [|1|])
      done
    done;
    
    let a = Tensor.cat [a1; a2] ~dim:0 in
    let b = Tensor.cat [m1.density; m2.density] ~dim:0 in
    
    (* Solve LP using interior point method *)
    let result = Optimization.InteriorPoint.solve
      ~objective:obj
      ~equality_constraints:a
      ~equality_rhs:b
      ~max_iter:1000
      ~tolerance:1e-8 in
    
    (* Compute optimal transport cost *)
    Tensor.dot (Tensor.reshape result [|-1|]) obj |>
    Tensor.float_value
end

module MOT = struct
  type t = {
    marginals: DiscreteMeasure.t array;
    cost: Tensor.t -> Tensor.t -> Tensor.t;
    epsilon: float;
  }

  let create marginals cost epsilon =
    if epsilon < 0.0 then
      failwith "Epsilon must be non-negative";
    {marginals; cost; epsilon}

  let evaluate prob plan =
    let supports = Array.map DiscreteMeasure.support prob.marginals in
    let cost_values = prob.cost supports.(0) supports.(Array.length supports - 1) in
    Tensor.dot (Tensor.reshape plan [|-1|]) (Tensor.reshape cost_values [|-1|]) |>
    Tensor.float_value

  let get_marginals prob = prob.marginals
  
  let get_dimension prob =
    (Tensor.shape (DiscreteMeasure.support prob.marginals.(0))).(1)
end

module Solver = struct
  type solver_params = {
    max_iter: int;
    learning_rate: float;
    tolerance: float;
    discretization_points: int;
  }

  let default_params = {
    max_iter = 1000;
    learning_rate = 0.01;
    tolerance = 1e-6;
    discretization_points = 100;
  }

  (* Project plan onto constraint set *)
  let project_constraints prob plan =
    let marginals = prob.MOT.marginals in
    let n = Array.length marginals in
    
    (* Project onto marginal constraints *)
    let proj = ref plan in
    for i = 0 to n-1 do
      let mu_i = DiscreteMeasure.density marginals.(i) |> Option.get in
      let sum_i = Tensor.sum !proj ~dim:[|i|] in
      let scale_i = Tensor.div mu_i sum_i in
      let expanded_scale = Tensor.view scale_i 
        (Array.make n 1 |> Array.mapi (fun j x -> if j = i then -1 else x)) in
      proj := Tensor.mul !proj expanded_scale
    done;
    
    (* Project onto martingale constraint *)
    for i = 0 to n-2 do
      let supports = Array.map DiscreteMeasure.support marginals in
      let curr_vals = Tensor.index_select supports.(i) ~dim:0 
        ~index:(Tensor.arange ~end_:((Tensor.shape !proj).(i))) in
      let next_vals = Tensor.index_select supports.(i+1) ~dim:0
        ~index:(Tensor.arange ~end_:((Tensor.shape !proj).(i+1))) in
      
      let diff = Tensor.sub next_vals curr_vals in
      let norm_diff = Tensor.norm diff ~p:2 ~dim:[|-1|] in
      let mask = Tensor.le norm_diff (Tensor.scalar_tensor prob.MOT.epsilon) in
      
      proj := Tensor.where mask !proj (Tensor.zeros_like !proj)
    done;
    
    !proj

  (* Main solver *)
  let solve ?(params=default_params) prob =
    let n = params.discretization_points in
    let dim = MOT.get_dimension prob in
    
    (* Initialize transport plan *)
    let plan = Tensor.ones [|n; n|] |>
      Tensor.div_scalar (float_of_int (n * n)) in
    
    let rec iterate plan iter =
      if iter >= params.max_iter then plan
      else
        (* Gradient step *)
        let cost_grad = Tensor.grad_of_fn
          (fun p -> MOT.evaluate prob p) plan in
        let new_plan = Tensor.sub plan 
          (Tensor.mul_scalar cost_grad params.learning_rate) in
        
        (* Project onto constraints *)
        let proj_plan = project_constraints prob new_plan in
        
        (* Check convergence *)
        let diff = Tensor.norm 
          (Tensor.sub proj_plan plan) 
          ~p:2 ~dim:[|0; 1|] |>
          Tensor.float_value in
        
        if diff < params.tolerance then proj_plan
        else iterate proj_plan (iter + 1)
    in
    
    iterate plan 0
end


type error_metrics = {
  constraint_violation: float;
  objective_gap: float;
  stability_measure: float;
  condition_number: float;
}

type convergence_constants = {
  lipschitz: float;
  moment_bound: float;
  support_bound: float option;
}

let compute_constraint_violation problem solution =
  let marginals = MOT.get_marginals problem in
  
  (* Marginal constraints *)
  let marginal_error = Array.map2
    (fun mu pi ->
      let diff = Tensor.sub
        (Tensor.sum pi ~dim:[|1|])
        (DiscreteMeasure.density mu |> Option.get) in
      Tensor.norm diff ~p:1 ~dim:[|0|] |>
      Tensor.float_value)
    marginals [|solution|] |>
  Array.fold_left max 0.0 in
  
  (* Martingale constraints *)
  let martingale_error =
    let n = Array.length marginals in
    Array.init (n-1) (fun i ->
      let curr_support = DiscreteMeasure.support marginals.(i) in
      let next_support = DiscreteMeasure.support marginals.(i+1) in
      let slice = Tensor.narrow solution ~dim:0 ~start:i ~length:2 in
      
      let conditional_exp = Tensor.sum 
        (Tensor.mul slice next_support) ~dim:[|1|] in
      let curr_val = Tensor.mul 
        (Tensor.narrow slice ~dim:0 ~start:0 ~length:1)
        curr_support in
      
      Tensor.norm 
        (Tensor.sub conditional_exp curr_val)
        ~p:1 ~dim:[|0|] |>
      Tensor.float_value) |>
    Array.fold_left max 0.0 in
  
  max marginal_error martingale_error

let compute_dual_bound problem =
  let marginals = MOT.get_marginals problem in
  let n = Array.length marginals in
  
  (* Initialize dual variables *)
  let dual_vars = Array.map (fun m ->
    Tensor.zeros [|Tensor.shape 
      (DiscreteMeasure.support m).(0)|]
  ) marginals in
  
  let rec iterate dual_vars iter =
    if iter >= 1000 then dual_vars
    else
      (* Update variables using subgradient method *)
      let gradients = Array.mapi (fun i var ->
        let support = DiscreteMeasure.support marginals.(i) in
        let density = DiscreteMeasure.density marginals.(i) |> 
          Option.get in
        
        let cost_contrib = if i = n-1 then
          MOT.evaluate problem support
        else Tensor.zeros_like var in
        
        Tensor.add
          (Tensor.sub (Tensor.exp var) density)
          cost_contrib
      ) dual_vars in
      
      let step_size = 1.0 /. sqrt (float_of_int (iter + 1)) in
      
      let new_vars = Array.map2
        (fun var grad ->
          Tensor.sub var 
            (Tensor.mul_scalar grad step_size))
        dual_vars gradients in
      
      let max_change = Array.map2
        (fun old_v new_v ->
          Tensor.max (Tensor.abs (Tensor.sub old_v new_v)) |>
          Tensor.float_value)
        dual_vars new_vars |>
        Array.fold_left max 0.0 in
      
      if max_change < 1e-6 then new_vars
      else iterate new_vars (iter + 1)
  in
  
  let optimal_duals = iterate dual_vars 0 in
  
  (* Dual objective *)
  Array.fold_left2
    (fun acc var mu ->
      acc +. (Tensor.dot var 
        (DiscreteMeasure.density mu |> Option.get) |>
       Tensor.float_value))
    0.0 optimal_duals marginals

let compute_condition_number solution =
  let s = Tensor.svd solution |> fun (_, s, _) -> s in
  let max_s = Tensor.max s |> Tensor.float_value in
  let min_s = Tensor.min s |> Tensor.float_value in
  if min_s < 1e-10 then infinity
  else max_s /. min_s

let compute_error_metrics problem solution =
  let constraint_viol = 
    compute_constraint_violation problem solution in
  let primal_obj = MOT.evaluate problem solution in
  let dual_bound = compute_dual_bound problem in
  let objective_gap = abs_float (primal_obj -. dual_bound) in
  
  (* Stability analysis *)
  let perturbed = Tensor.add solution
    (Tensor.mul_scalar 
       (Tensor.randn_like solution) 1e-6) in
  let perturbed_obj = MOT.evaluate problem perturbed in
  let stability = abs_float (perturbed_obj -. primal_obj) /. 
    1e-6 in
  
  let cond_num = compute_condition_number solution in
  
  {
    constraint_violation = constraint_viol;
    objective_gap;
    stability_measure = stability;
    condition_number = cond_num;
  }

let analyze_stability problem eps_array =
  Array.map (fun eps ->
    let perturbed_marginals = 
      Array.map (fun m ->
        let support = DiscreteMeasure.support m in
        let density = DiscreteMeasure.density m |> Option.get in
        let noise = Tensor.mul_scalar 
          (Tensor.randn_like support) eps in
        DiscreteMeasure.create density
          ~support:(Tensor.add support noise))
      (MOT.get_marginals problem) in
    
    let perturbed_problem = {problem with
      MOT.marginals = perturbed_marginals} in
    let solution = Solver.solve perturbed_problem in
    
    let metrics = compute_error_metrics 
      perturbed_problem solution in
    (eps, metrics))
  eps_array

let estimate_error_bounds problem solution =
  let metrics = compute_error_metrics problem solution in
  
  let lipschitz_const = 
    metrics.stability_measure /. 
    metrics.constraint_violation in
  
  let error_bound = 
    lipschitz_const *. 
    (metrics.constraint_violation +. 
     sqrt metrics.objective_gap) in
  
  let std_dev = sqrt metrics.stability_measure in
  let margin = 1.96 *. std_dev in  (* 95% confidence *)
  let ci = (error_bound -. margin, error_bound +. margin) in
  
  (error_bound, ci)

let compute_convergence_rate constants n epsilon =
  match constants.support_bound with
  | Some b ->
      (* Bounded support case *)
      let c = constants.lipschitz *. b in
      c *. (epsilon +. 1.0 /. sqrt (float_of_int n))
  | None ->
      (* Unbounded support case *)
      let theta = 2.0 in  (* moment parameter *)
      let term1 = epsilon *. 
        (1.0 +. constants.moment_bound) in
      let term2 = constants.moment_bound /. 
        (float_of_int n ** (1.0 -. 1.0 /. theta)) in
      constants.lipschitz *. (term1 +. term2)