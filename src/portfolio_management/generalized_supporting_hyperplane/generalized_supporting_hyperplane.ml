open Torch

type optimization_params = {
  max_iter: int;
  tolerance: float;
  learning_rate: float;
  momentum: float;
}

type tensor_pair = {
  first: Tensor.t;
  second: Tensor.t;
}

module AssetHoldings = struct
  type state = {
    value: float;
    weights: Tensor.t;
    returns: Tensor.t;
    costs: Tensor.t;
  }

  let compute_next_value value weights prev_weights returns costs =
    let portfolio_return = 
      Tensor.matmul weights (Tensor.unsqueeze returns 1)
      |> Tensor.squeeze
      |> Tensor.item in
    
    let adjustment_costs = 
      Tensor.abs (Tensor.sub weights prev_weights)
      |> Tensor.mul costs
      |> Tensor.sum
      |> Tensor.item in
    
    value *. (1. +. portfolio_return) *. (1. -. adjustment_costs)

  let update_state state prev_weights =
    let next_value = 
      compute_next_value 
        state.value state.weights prev_weights state.returns state.costs in
    { state with value = next_value }

  let compute_stability state prev_weights min_return max_return =
    let portfolio_return = 
      Tensor.matmul state.weights (Tensor.unsqueeze state.returns 1)
      |> Tensor.squeeze in
    
    let adjustment_costs =
      Tensor.abs (Tensor.sub state.weights prev_weights)
      |> Tensor.mul state.costs
      |> Tensor.sum in
    
    let minimum_case_return = 
      Tensor.add portfolio_return (Tensor.neg adjustment_costs)
      |> Tensor.min
      |> Tensor.item in
    
    if minimum_case_return >= -1.0 then 1.0 else 0.0
end

(* Trading Constraints *)
module TradingConstraints = struct
  type t = {
    leverage_limit: float;           (* L ≥ 1 *)
    adjustment_limit: float;           (* Maximum adjustment *)
    holding_limits: Tensor.t;        (* Individual position limits *)
    stability_threshold: float;       (* Minimum stability probability *)
  }

  (* Leverage constraint *)
  let check_leverage weights max_leverage =
    let sum_abs = Tensor.sum (Tensor.abs weights) |> Tensor.item in
    sum_abs <= max_leverage

  (* Holding constraints *)
  let check_holdings weights limits =
    Tensor.all (Tensor.le (Tensor.abs weights) limits)

  (* Stability conditions *)
  let check_stability weights returns costs prev_weights =
    let long_positions = Tensor.relu weights in
    let short_positions = Tensor.neg (Tensor.relu (Tensor.neg weights)) in
    
    let min_return = Tensor.min returns |> Tensor.item in
    let max_return = Tensor.max returns |> Tensor.item in
    
    let stability_cond1 = 
      let long_exposure = Tensor.sum long_positions |> Tensor.item in
      let short_exposure = Tensor.sum short_positions |> Tensor.item in
      long_exposure *. min_return -. short_exposure *. max_return <= 1. in
    
    let stability_cond2 =
      let adjustment_costs = 
        Tensor.abs (Tensor.sub weights prev_weights)
        |> Tensor.mul costs
        |> Tensor.sum
        |> Tensor.item in
      adjustment_costs <= 1. in
    
    stability_cond1 && stability_cond2

  (* Check all constraints *)
  let check_constraints t weights prev_weights returns costs =
    check_leverage weights t.leverage_limit &&
    check_holdings weights t.holding_limits &&
    check_stability weights returns costs prev_weights

  (* Project weights onto feasible set *)
  let project t weights prev_weights =
    let projected = Tensor.copy weights in
    
    (* Project onto leverage constraint *)
    let leverage = Tensor.sum (Tensor.abs projected) |> Tensor.item in
    if leverage > t.leverage_limit then
      Tensor.mul_scalar_ projected (t.leverage_limit /. leverage);
    
    (* Project onto holding limits *)
    Tensor.clamp_ projected 
      (Tensor.neg t.holding_limits) 
      t.holding_limits;
    
    (* Project onto adjustment constraint *)
    let adjustment = 
      Tensor.sum (Tensor.abs (Tensor.sub projected prev_weights))
      |> Tensor.item in
    if adjustment > t.adjustment_limit then begin
      let scale = t.adjustment_limit /. adjustment in
      let diff = Tensor.sub projected prev_weights in
      Tensor.mul_scalar_ diff scale;
      Tensor.add_ projected prev_weights
    end;
    
    projected
end

(* Distributional Robust Optimization *)
module DistributionalRobust = struct
  type confidence_set = {
    a0: Tensor.t;  (* Equality constraints matrix *)
    d0: Tensor.t;  (* Equality constraints vector *)
    a1: Tensor.t;  (* Inequality constraints matrix *)
    d1: Tensor.t;  (* Inequality constraints vector *)
    support: tensor_pair;  (* Return support bounds *)
  }

  (* Create moment-based confidence set *)
  let create_moment_based returns confidence_level =
    let mean = Tensor.mean returns 0 in
    let centered = Tensor.sub returns (Tensor.unsqueeze mean 0) in
    let covariance = 
      Tensor.matmul 
        (Tensor.transpose centered 0 1)
        centered in
    
    let n = Tensor.size returns 1 in
    
    (* Mean constraints: E[r] = μ *)
    let a0 = Tensor.eye n in
    let d0 = mean in
    
    (* Covariance constraints: E[(r-μ)(r-μ)^T] ≤ Σ *)
    let scale = sqrt (confidence_level /. (1. -. confidence_level)) in
    let a1 = Tensor.mul_scalar covariance scale in
    let d1 = covariance in
    
    (* Support bounds *)
    let lower = Tensor.min returns 0 in
    let upper = Tensor.max returns 0 in
    
    { a0; d0; a1; d1; 
      support = { first = lower; second = upper } }

  (* Running objective *)
  let compute_objective utility weights returns costs prev_weights =
    let portfolio_return = 
      Tensor.matmul weights (Tensor.unsqueeze returns 1)
      |> Tensor.squeeze in
    
    let adjustment_costs = 
      Tensor.abs (Tensor.sub weights prev_weights)
      |> Tensor.mul costs
      |> Tensor.sum in
    
    let growth_rate = 
      Tensor.add (Tensor.ones_like portfolio_return) portfolio_return
      |> fun x -> Tensor.mul x (Tensor.sub (Tensor.ones_like adjustment_costs) adjustment_costs) in
    
    utility growth_rate |> Tensor.mean |> Tensor.item

  (* Solve minimum-case distribution problem *)
  let solve_minimum_case set weights =
    let n = Tensor.size weights 0 in
    let vars = Tensor.zeros [|n + 2|] in
    
    (* Form and solve linear program *)
    let obj = Tensor.cat [weights; Tensor.zeros [|2|]] 0 in
    
    let a_eq = Tensor.cat [set.a0; Tensor.zeros [|2; n|]] 1 in
    let b_eq = set.d0 in
    
    let a_ineq = Tensor.cat [set.a1; Tensor.eye 2] 1 in
    let b_ineq = set.d1 in
    
    (* Solve using matrix operations *)
    let result = 
      let kkt_matrix = 
        Tensor.cat [
          Tensor.cat [Tensor.eye n; Tensor.transpose a_eq 0 1] 1;
          Tensor.cat [a_eq; Tensor.zeros [|n; n|]] 1
        ] 0 in
      
      let kkt_rhs = 
        Tensor.cat [Tensor.neg obj; b_eq] 0 in
      
      let sol = 
        Tensor.solve kkt_matrix kkt_rhs in
      
      Tensor.narrow sol 0 0 n in
    
    result
end

(* Utility function module *)
module SeparableUtility = struct
  type t = {
    phi1: Tensor.t -> Tensor.t;
    phi2: Tensor.t -> Tensor.t;
    phi1_derivative: Tensor.t -> Tensor.t;
    phi2_derivative: Tensor.t -> Tensor.t;
    phi1_second_derivative: Tensor.t -> Tensor.t;
    phi2_second_derivative: Tensor.t -> Tensor.t;
    alpha: float;
    beta: float;
  }

  (* Log utility *)
  let create_log ~alpha ~beta = {
    phi1 = (fun x -> Tensor.log (Tensor.add x (Tensor.ones_like x)));
    phi2 = (fun c -> Tensor.log (Tensor.sub (Tensor.ones_like c) c));
    phi1_derivative = (fun x -> 
      Tensor.div (Tensor.ones_like x) (Tensor.add x (Tensor.ones_like x)));
    phi2_derivative = (fun c -> 
      Tensor.div (Tensor.neg (Tensor.ones_like c)) (Tensor.sub (Tensor.ones_like c) c));
    phi1_second_derivative = (fun x ->
      let denom = Tensor.add x (Tensor.ones_like x) in
      Tensor.div (Tensor.neg (Tensor.ones_like x)) (Tensor.mul denom denom));
    phi2_second_derivative = (fun c ->
      let denom = Tensor.sub (Tensor.ones_like c) c in
      Tensor.div (Tensor.neg (Tensor.ones_like c)) (Tensor.mul denom denom));
    alpha;
    beta;
  }

  (* Power utility *)
  let create_power ~alpha ~beta ~gamma = {
    phi1 = (fun x -> 
      Tensor.pow (Tensor.add x (Tensor.ones_like x)) (Tensor.float_vec [|gamma|]));
    phi2 = (fun c ->
      Tensor.pow (Tensor.sub (Tensor.ones_like c) c) (Tensor.float_vec [|gamma|]));
    phi1_derivative = (fun x ->
      let base = Tensor.add x (Tensor.ones_like x) in
      Tensor.mul (Tensor.float_vec [|gamma|]) 
        (Tensor.pow base (Tensor.float_vec [|gamma -. 1.|])));
    phi2_derivative = (fun c ->
      let base = Tensor.sub (Tensor.ones_like c) c in
      Tensor.mul (Tensor.float_vec [|gamma|])
        (Tensor.pow base (Tensor.float_vec [|gamma -. 1.|])));
    phi1_second_derivative = (fun x ->
      let base = Tensor.add x (Tensor.ones_like x) in
      Tensor.mul 
        (Tensor.float_vec [|gamma *. (gamma -. 1.)|])
        (Tensor.pow base (Tensor.float_vec [|gamma -. 2.|])));
    phi2_second_derivative = (fun c ->
      let base = Tensor.sub (Tensor.ones_like c) c in
      Tensor.mul
        (Tensor.float_vec [|gamma *. (gamma -. 1.)|])
        (Tensor.pow base (Tensor.float_vec [|gamma -. 2.|])));
    alpha;
    beta;
  }

  let evaluate t returns costs =
    Tensor.add
      (Tensor.mul_scalar (t.phi1 returns) t.alpha)
      (Tensor.mul_scalar (t.phi2 costs) t.beta)
end

(* Hyperplane approximation module *)
module Hyperplane = struct
  type t = {
    a: float;        (* Return component slope *)
    b: float;        (* Cost component slope *)
    gamma: float;    (* Intercept *)
    x_point: float;  (* Partition point for returns *)
    c_point: float;  (* Partition point for costs *)
  }

  (* Auxiliary function computation *)
  let auxiliary_function utility x c =
    let open Tensor in
    let ret = add (ones_like x) x in
    let cost = sub (ones_like c) c in
    utility ret cost

  (* Compute hyperplane coefficients *)
  let create utility x_point c_point =
    let x = Tensor.float_vec [|x_point|] in
    let c = Tensor.float_vec [|c_point|] in
    
    (* Compute derivatives *)
    let deriv_x = 
      Tensor.gradient (fun x -> auxiliary_function utility x c) x in
    let deriv_c = 
      Tensor.gradient (fun c -> auxiliary_function utility x c) c in
    
    let a = Tensor.item (Tensor.get deriv_x [|0|]) in
    let b = Tensor.item (Tensor.get deriv_c [|0|]) in
    
    let f_val = Tensor.item (auxiliary_function utility x c) in
    let gamma = f_val -. a *. x_point -. b *. c_point in
    
    { a; b; gamma; x_point; c_point }

  (* Evaluate hyperplane at point *)
  let evaluate t x c =
    t.a *. x +. t.b *. c +. t.gamma
end

(* Reliability analysis module *)
module Reliability = struct
  type reliability_components = {
    reliability_x: float;
    reliability_c: float;
    total_reliability: float;
  }

  let analyze_reliability_monotonicity utility x c x_partition c_partition =
    let tensor_x = Tensor.float_vec [|x|] in
    let tensor_c = Tensor.float_vec [|c|] in
    let tensor_x_part = Tensor.float_vec [|x_partition|] in
    let tensor_c_part = Tensor.float_vec [|c_partition|] in
    
    (* Compute derivatives *)
    let dx = Tensor.item (utility.SeparableUtility.phi1_derivative tensor_x) in
    let dc = Tensor.item (utility.phi2_derivative tensor_c) in
    let dx_part = Tensor.item (utility.phi1_derivative tensor_x_part) in
    let dc_part = Tensor.item (utility.phi2_derivative tensor_c_part) in
    
    (* Check monotonicity conditions *)
    let x_increasing = dx > dx_part in
    let c_increasing = dc > dc_part in
    
    (x_increasing, c_increasing)

  let compute_max_reliability utility hyperplane x_range c_range num_points =
    let x_points = Array.init num_points (fun i ->
      x_range.(0) +. (x_range.(1) -. x_range.(0)) *. 
      float_of_int i /. float_of_int (num_points - 1)) in
    let c_points = Array.init num_points (fun i ->
      c_range.(0) +. (c_range.(1) -. c_range.(0)) *. 
      float_of_int i /. float_of_int (num_points - 1)) in
    
    let compute_point_reliability x c =
      let tensor_x = Tensor.float_vec [|x|] in
      let tensor_c = Tensor.float_vec [|c|] in
      
      (* True function value *)
      let true_val = 
        SeparableUtility.evaluate utility tensor_x tensor_c
        |> Tensor.item in
      
      (* Hyperplane approximation *)
      let approx_val = Hyperplane.evaluate hyperplane x c in
      
      abs_float (true_val -. approx_val) in
    
    let reliabilitys = ref [] in
    Array.iter (fun x ->
      Array.iter (fun c ->
        reliabilitys := compute_point_reliability x c :: !reliabilitys
      ) c_points
    ) x_points;
    
    let max_reliability = List.fold_left max neg_infinity !reliabilitys in
    let reliability_x = 
      Array.fold_left (fun acc x -> 
        max acc (compute_point_reliability x c_range.(0)))
        neg_infinity x_points in
    let reliability_c =
      Array.fold_left (fun acc c ->
        max acc (compute_point_reliability x_range.(0) c))
        neg_infinity c_points in
    
    { reliability_x; reliability_c; total_reliability = max_reliability }
end

(* Partition refinement module *)
module PartitionRefinement = struct
  type partition_point = {
    x: float;
    c: float;
    reliability: float;
  }

  let compute_successive_points utility current_point reliability_tolerance =
    let x = current_point.x in
    let c = current_point.c in
    
    (* Solve for A* *)
    let solve_a_equation a =
      let tensor_x = Tensor.float_vec [|x +. a|] in
      let tensor_c = Tensor.float_vec [|c|] in
      let phi_val = utility.SeparableUtility.phi1 tensor_x |> Tensor.item in
      let phi_deriv = utility.phi1_derivative tensor_x |> Tensor.item in
      phi_deriv *. a -. phi_val +. utility.alpha *. reliability_tolerance in
    
    (* Solve for D* *)
    let solve_d_equation d =
      let tensor_x = Tensor.float_vec [|x|] in
      let tensor_c = Tensor.float_vec [|c +. d|] in
      let phi_val = utility.SeparableUtility.phi2 tensor_c |> Tensor.item in
      let phi_deriv = utility.phi2_derivative tensor_c |> Tensor.item in
      phi_deriv *. d -. phi_val +. utility.beta *. reliability_tolerance in
    
    (* Binary search *)
    let rec binary_search f a b eps =
      let mid = (a +. b) /. 2. in
      let f_mid = f mid in
      if abs_float f_mid < eps then mid
      else if f_mid *. f a < 0. then binary_search f a mid eps
      else binary_search f mid b eps in
    
    let a_star = binary_search solve_a_equation 0. 1. 1e-6 in
    let d_star = binary_search solve_d_equation 0. 1. 1e-6 in
    
    let next_x = x +. a_star in
    let next_c = c +. d_star in
    
    { x = next_x; 
      c = next_c; 
      reliability = reliability_tolerance }

  let compute_log_utility_points ~current_x ~current_c ~reliability_x ~reliability_c =
    let ax = exp reliability_x -. 1.0 in
    let dc = 1.0 -. exp (-.reliability_c) in
    
    let next_x = current_x *. (1.0 +. ax) +. ax in
    let next_c = current_c *. (1.0 -. dc) +. dc in
    
    { x = next_x;
      c = next_c;
      reliability = (reliability_x +. reliability_c) /. 2. }
end

(* Complete optimization system *)
module RobustPortfolio = struct
  type t = {
    utility: SeparableUtility.t;
    hyperplanes: Hyperplane.t array;
    partitions: PartitionRefinement.partition_point array;
    constraints: TradingConstraints.t;
    confidence: DistributionalRobust.confidence_set;
  }

  let create ~returns ~confidence_level ~constraints ~utility_type ~alpha ~beta ~reliability_tol =
    let utility = match utility_type with
      | `Log -> SeparableUtility.create_log ~alpha ~beta
      | `Power gamma -> SeparableUtility.create_power ~alpha ~beta ~gamma in
    
    let confidence = 
      DistributionalRobust.create_moment_based returns confidence_level in
    
    let x_range = [|
      Tensor.min confidence.support.first |> Tensor.item;
      Tensor.max confidence.support.second |> Tensor.item
    |] in
    let c_range = [|0.; constraints.adjustment_limit|] in
    
    let initial_partition = {
      PartitionRefinement.x = x_range.(0);
      c = c_range.(0);
      reliability = reliability_tol
    } in
    
    let partition_points = 
      Array.make 1 initial_partition in
    
    let hyperplanes = 
      Array.map (fun p ->
        Hyperplane.create utility p.x p.c)
        partition_points in
    
    { utility; hyperplanes; partitions = partition_points; 
      constraints; confidence }

  let optimize t weights prev_weights =
    let minimum_case = 
      DistributionalRobust.solve_minimum_case t.confidence weights in
    
    let objectives = Array.map (fun h ->
      let returns = 
        Tensor.matmul weights t.confidence.support.first in
      let costs =
        Tensor.abs (Tensor.sub weights prev_weights) in
      Hyperplane.evaluate h 
        (Tensor.item returns) 
        (Tensor.item costs))
      t.hyperplanes in
    
    let min_obj = Array.fold_left min infinity objectives in
    
    let projected = 
      TradingConstraints.project t.constraints weights prev_weights in
    
    if TradingConstraints.check_constraints 
         t.constraints projected prev_weights 
         t.confidence.support.first
         (Tensor.ones [|Tensor.size weights 0|]) then
      projected
    else
      prev_weights
end