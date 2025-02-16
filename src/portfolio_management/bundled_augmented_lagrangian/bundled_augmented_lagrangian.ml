open Torch

type constraints = {
  c: Tensor.t;  (* Objective vector *)
  a: Tensor.t;  (* Constraint matrix *)
  b: Tensor.t;  (* Constraint vector *)
  n: int;       (* Number of variables *)
  m: int;       (* Number of constraints *)
}

type solution = {
  x: Tensor.t;  (* Primal solution *)
  y: Tensor.t;  (* Dual solution *)
  obj_val: float;  (* Objective value *)
  primal_feas: float;  (* Primal feasibility *)
  dual_feas: float;    (* Dual feasibility *)
}

type params = {
  rho: float;      (* Augmented Lagrangian parameter *)
  beta: float;     (* Bundle method parameter *)
  max_iter: int;   (* Maximum iterations *)
  tol: float;      (* Tolerance for convergence *)
}

let affine_map a x b =
  Tensor.(mm a x - b)

let compute_augmented_lagragian prob x y rho =
  let ax_b = affine_map prob.a x prob.b in
  let lin_term = Tensor.dot prob.c x in
  let dual_term = Tensor.dot y ax_b in
  let quad_term = Tensor.(mul_scalar (Tensor.dot ax_b ax_b) (rho /. 2.)) in
  Tensor.(add (add lin_term dual_term) quad_term)

let compute_grad_augmented_lagragian prob x y rho =
  let ax_b = affine_map prob.a x prob.b in
  Tensor.(
    add prob.c
      (add 
        (mm (transpose prob.a 0 1) y)
        (mul_scalar (mm (transpose prob.a 0 1) ax_b) rho)
      )
  )

let compute_min_augmented_lagragian prob y rho =
  (* Compute minimum of Lagrangian over x ∈ Ω *)
  let grad = compute_grad_augmented_lagragian prob 
    (Tensor.zeros [prob.n; 1]) y rho in
  Tensor.min grad

(* Bundle management *)
module Bundle = struct
  type point = {
    x: Tensor.t;
    g: Tensor.t;  (* subgradient *)
    fx: float;
    age: int;
  }

  type t = {
    points: point list;
    max_size: int;
    aggregated_g: Tensor.t option;
  }

  let create max_size = {
    points = [];
    max_size;
    aggregated_g = None;
  }

  let add_point bundle x g fx =
    let point = { x; g; fx; age = 0 } in
    { bundle with 
      points = point :: bundle.points;
      aggregated_g = Some g }

  let size bundle = List.length bundle.points
end

(* Inner approximation *)
module InnerApprox = struct
  type approx_set = {
    points: Tensor.t list;
    weights: Tensor.t;
  }

  let project_onto_set set x =
    let n = List.length set.points in
    let points_tensor = Tensor.stack (Array.of_list set.points) 0 in
    let weights = Tensor.softmax set.weights (-1) in
    Tensor.mm weights points_tensor

  let create_line_segment v w = {
    points = [v; w];
    weights = Tensor.ones [2]
  }
end

(* Core convergence tracking *)
module Convergence = struct
  type metrics = {
    primal_residual: float;
    dual_residual: float;
    cost_gap: float;
    num_descent_steps: int;
    num_null_steps: int;
  }

  let compute_metrics state =
    let primal_res = Tensor.float_value (
      Tensor.norm (affine_map state.Bala.prob.a state.Bala.x state.Bala.prob.b)
    ) in
    
    let dual_res = Tensor.float_value (
      Tensor.norm (compute_grad_augmented_lagragian 
        state.Bala.prob state.Bala.x state.Bala.y state.Bala.rho)
    ) in
    
    let obj_val = Tensor.float_value (
      Tensor.dot state.Bala.prob.c state.Bala.x
    ) in
    
    let dual_val = Tensor.float_value (
      compute_dual_value state.Bala.prob state.Bala.y
    ) in
    
    { primal_residual = primal_res;
      dual_residual = dual_res;
      cost_gap = abs_float (obj_val -. dual_val);
      num_descent_steps = state.Bala.descent_steps;
      num_null_steps = state.Bala.null_steps }
end

let compute_dual_value prob y =
  let min_augmented_lagragian = compute_min_augmented_lagragian prob y 0.0 in
  Tensor.add min_augmented_lagragian (Tensor.dot prob.b y)

let compute_dual_gradient prob x =
  affine_map prob.a x prob.b

(* Bundle augmented augmented_lagragianrangian algorithm *)
module Bala = struct
  type state = {
    prob: constraints;
    x: Tensor.t;           (* Current primal solution *)
    y: Tensor.t;           (* Current dual solution *)
    omega: InnerApprox.approx_set;  (* Current inner approximation *)
    bundle: Bundle.t;      (* Current bundle *)
    rho: float;           (* Current penalty parameter *)
    beta: float;          (* Current descent parameter *)
    iter: int;            (* Iteration counter *)
    descent_steps: int;    (* Number of descent steps *)
    null_steps: int;       (* Number of null steps *)
    history: metrics list;  (* Convergence history *)
  }

  let create_initial_state prob params = {
    prob;
    x = Tensor.zeros [prob.n; 1];
    y = Tensor.zeros [prob.m; 1];
    omega = InnerApprox.create_line_segment 
      (Tensor.zeros [prob.n; 1]) 
      (Tensor.zeros [prob.n; 1]);
    bundle = Bundle.create 20;
    rho = params.rho;
    beta = params.beta;
    iter = 0;
    descent_steps = 0;
    null_steps = 0;
    history = [];
  }
end

let solve_approx_subproblem prob omega y rho =
  let objective x = 
    compute_augmented_lagragian prob x y rho in
  
  let gradient x =
    compute_grad_augmented_lagragian prob x y rho in
  
  (* Projected gradient descent *)
  let max_iter = 100 in
  let alpha = 0.01 in
  let rec iterate x iter =
    if iter >= max_iter then x
    else begin
      let grad = gradient x in
      let x_new = Tensor.sub x (Tensor.mul_scalar grad alpha) in
      let x_proj = InnerApprox.project_onto_set omega x_new in
      if Tensor.float_value (Tensor.norm (Tensor.sub x_proj x)) < 1e-8 then x_proj
      else iterate x_proj (iter + 1)
    end
  in
  
  let x0 = InnerApprox.project_onto_set omega 
    (Tensor.zeros [prob.n; 1]) in
  iterate x0 0

(* Step size control *)
module StepSize = struct
  type step_params = {
    min_step: float;
    max_step: float;
    increase_factor: float;
    decrease_factor: float;
  }

  let create_default_params () = {
    min_step = 1e-8;
    max_step = 1e8;
    increase_factor = 2.0;
    decrease_factor = 0.5;
  }

  let compute_step_size state progress params =
    let current = Tensor.float_value (
      Tensor.norm (Tensor.sub state.Bala.y progress.Bala.y)
    ) in
    
    if progress.Convergence.num_descent_steps > 
       progress.Convergence.num_null_steps then
      min params.max_step (current *. params.increase_factor)
    else
      max params.min_step (current *. params.decrease_factor)
end

let update_bundle state wk yk is_descent =
  let bundle = state.Bala.bundle in
  let points = 
    if List.length bundle.Bundle.points >= 20 then
      List.take 15 bundle.Bundle.points
    else bundle.Bundle.points in
  
  let new_point = { Bundle.
    x = wk;
    g = compute_grad_augmented_lagragian state.Bala.prob wk yk state.Bala.rho;
    fx = Tensor.float_value (
      compute_augmented_lagragian state.Bala.prob wk yk state.Bala.rho
    );
    age = 0
  } in
  
  let aged_points = List.map (fun p ->
    {p with Bundle.age = p.Bundle.age + 1}
  ) points in
  
  { bundle with Bundle.points = new_point :: aged_points }

(* Quadratic growth *)
module QuadraticGrowth = struct
  type growth_certificate = {
    alpha: float;
    valid_radius: float;
    verified_points: (Tensor.t * float) list;
    condition_satisfied: bool;
  }

  let verify_growth_condition g y y_star alpha =
    let dist = Tensor.norm (Tensor.sub y y_star) in
    let g_diff = Tensor.(float_value (sub (g y) (g y_star))) in
    g_diff >= alpha *. (Tensor.float_value dist) ** 2.0

  let estimate_growth_parameter g points max_radius =
    let rec binary_search low high tol iter max_iter =
      if iter >= max_iter || high -. low < tol then
        (low +. high) /. 2.0
      else
        let mid = (low +. high) /. 2.0 in
        let valid = List.for_all (fun p1 ->
          List.for_all (fun p2 ->
            let dist = Tensor.float_value (Tensor.norm (Tensor.sub p1 p2)) in
            if dist <= max_radius then
              verify_growth_condition g p1 p2 mid
            else true
          ) points
        ) points in
        
        if valid then
          binary_search mid high tol (iter + 1) max_iter
        else
          binary_search low mid tol (iter + 1) max_iter
    in
    
    binary_search 0.0 1.0 1e-6 0 100
end

(* Rate adjustment *)
module RateAdjustment = struct
  type rate_info = {
    current_rate: float;
    target_rate: float option;
    stable: bool;
    suggested_params: float * float;  (* rho, beta *)
  }

  let compute_local_rates history window_size =
    let recent = List.take window_size history in
    List.map2 (fun s1 s2 ->
      let p1 = Tensor.float_value (
        Tensor.norm (affine_map s1.Bala.prob.a s1.Bala.x s1.Bala.prob.b)
      ) in
      let p2 = Tensor.float_value (
        Tensor.norm (affine_map s2.Bala.prob.a s2.Bala.x s2.Bala.prob.b)
      ) in
      if p1 > 0.0 then p2 /. p1 else 1.0
    ) (List.tl recent) recent

  let adjust_rates state history =
    let rates = compute_local_rates history 5 in
    let current_rate = 
      match rates with
      | [] -> 1.0
      | rs -> List.fold_left (+.) 0.0 rs /. float_of_int (List.length rs) in
    
    let stability = 
      let variance = List.fold_left (fun acc r ->
        acc +. (r -. current_rate) ** 2.0
      ) 0.0 rates /. float_of_int (List.length rates) in
      variance < 0.1 in
    
    let growth_cert = 
      QuadraticGrowth.estimate_growth_parameter
        (compute_dual_value state.Bala.prob)
        (List.map (fun s -> s.Bala.y) history)
        1.0 in
    
    let target_rate =
      Some (1.0 -. min state.Bala.beta (growth_cert /. state.Bala.rho)) in
    
    let rho, beta =
      match target_rate with
      | Some target when stability ->
          if current_rate > target +. 0.1 then
            (state.Bala.rho *. 1.5, state.Bala.beta *. 0.9)
          else if current_rate < target -. 0.1 then
            (state.Bala.rho *. 0.8, state.Bala.beta *. 1.1)
          else
            (state.Bala.rho, state.Bala.beta)
      | _ ->
          if stability then
            (state.Bala.rho, state.Bala.beta)
          else
            (state.Bala.rho *. 1.2, state.Bala.beta *. 0.9) in
    
    { current_rate;
      target_rate;
      stable = stability;
      suggested_params = (rho, beta) }
end

(* Solver *)
module Solver = struct
  type solver_state = {
    iteration: int;
    bala_state: Bala.state;
    avg_state: AverageIterateAnalysis.average_state;
    best_solution: solution option;
    convergence_history: Convergence.metrics list;
  }

  let create_solver_state prob params = {
    iteration = 0;
    bala_state = Bala.create_initial_state prob params;
    avg_state = AverageIterateAnalysis.create_average_state prob;
    best_solution = None;
    convergence_history = [];
  }

  let update_best_solution state metrics =
    match state.best_solution with
    | None -> 
        Some { 
          x = state.bala_state.Bala.x;
          y = state.bala_state.Bala.y;
          obj_val = Tensor.float_value (
            Tensor.dot state.bala_state.Bala.prob.c state.bala_state.Bala.x
          );
          primal_feas = metrics.Convergence.primal_residual;
          dual_feas = metrics.Convergence.dual_residual
        }
    | Some best ->
        if metrics.Convergence.primal_residual < best.primal_feas &&
           metrics.Convergence.dual_residual < best.dual_feas then
          Some { 
            x = state.bala_state.Bala.x;
            y = state.bala_state.Bala.y;
            obj_val = Tensor.float_value (
              Tensor.dot state.bala_state.Bala.prob.c state.bala_state.Bala.x
            );
            primal_feas = metrics.Convergence.primal_residual;
            dual_feas = metrics.Convergence.dual_residual
          }
        else Some best

  let solve_iteration state params =
    let bala = state.bala_state in
    
    (* Solve approximated subproblem *)
    let wk = solve_approx_subproblem 
      bala.prob bala.omega bala.y bala.rho in
    
    (* Compute dual candidate *)
    let ax_b = affine_map bala.prob.a wk bala.prob.b in
    let zk = Tensor.add bala.y (Tensor.mul_scalar ax_b bala.rho) in
    
    (* Test descent condition *)
    let g_yk = compute_augmented_lagragian bala.prob bala.x bala.y bala.rho in
    let g_zk = compute_augmented_lagragian bala.prob wk zk bala.rho in
    let gk_zk = compute_augmented_lagragian bala.prob wk bala.y bala.rho in
    
    let is_descent = Tensor.float_value (
      Tensor.sub g_yk g_zk
    ) >= bala.beta *. Tensor.float_value (
      Tensor.sub g_yk gk_zk
    ) in
    
    let new_bala =
      if is_descent then
        (* Descent step *)
        let new_omega = InnerApprox.create_line_segment wk zk in
        let new_bundle = update_bundle bala wk zk true in
        { bala with
          x = wk;
          y = zk;
          omega = new_omega;
          bundle = new_bundle;
          descent_steps = bala.descent_steps + 1 }
      else
        (* Null step *)
        let new_omega = InnerApprox.create_line_segment bala.x wk in
        let new_bundle = update_bundle bala wk bala.y false in
        { bala with
          omega = new_omega;
          bundle = new_bundle;
          null_steps = bala.null_steps + 1 } in
    
    (* Update average state *)
    let new_avg = AverageIterateAnalysis.update_average_state 
      new_bala state.avg_state is_descent in
    
    (* Compute convergence metrics *)
    let metrics = Convergence.compute_metrics new_bala in
    
    (* Adjust rates if needed *)
    let rate_info = RateAdjustment.adjust_rates new_bala 
      (metrics :: state.convergence_history) in
    let rho, beta = rate_info.suggested_params in
    
    let final_bala = { new_bala with
      rho = max 0.1 (min 1e6 rho);
      beta = max 0.1 (min 0.9 beta) } in
    
    { iteration = state.iteration + 1;
      bala_state = final_bala;
      avg_state = new_avg;
      best_solution = update_best_solution state metrics;
      convergence_history = metrics :: state.convergence_history }

  let check_termination state params =
    let metrics = List.hd state.convergence_history in
    metrics.Convergence.primal_residual < params.tol &&
    metrics.Convergence.dual_residual < params.tol &&
    metrics.Convergence.cost_gap < params.tol

  let solve prob params =
    let state = ref (create_solver_state prob params) in
    let converged = ref false in
    
    while not !converged && !state.iteration < params.max_iter do
      state := solve_iteration !state params;
      converged := check_termination !state params
    done;
end