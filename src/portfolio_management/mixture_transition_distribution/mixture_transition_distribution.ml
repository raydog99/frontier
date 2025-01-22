open Torch

type node_id = int
type weight = float

type edge = {
  source: node_id;
  target: node_id;
  weight: weight;
}

type network = {
  nodes: node_id array;
  adj_matrix: Tensor.t;
  weights_matrix: Tensor.t;
  properties: network_properties;
}

type network_properties = {
  in_degree_distribution: int array;
  out_degree_distribution: int array;
  weight_distribution: int array;
  clustering_coefficients: Tensor.t;
}

type assortativity_measures = {
  in_in: float;
  in_out: float;
  out_in: float;
  out_out: float;
  total: float;
}

(* Markov property and state space handling *)
module MarkovProperty = struct
  type state_history = {
    current: Tensor.t;
    past: Tensor.t list;
    max_order: int;
  }

  (* Create state history tracker *)
  let create_history ~max_order ~n_series =
    {
      current = Tensor.zeros [|n_series|];
      past = [];
      max_order = max_order;
    }

  (* Update state history *)
  let update_history history new_state =
    let updated_past = 
      history.current :: 
      (List.take (history.max_order - 1) history.past) in
    { history with
      current = new_state;
      past = updated_past;
    }

  (* Check Markov property *)
  let check_markov_property history state_probs future_state =
    let n_series = Tensor.size history.current 0 in
    let cond_prob = Tensor.zeros [|n_series|] in
    
    for j = 0 to n_series - 1 do
      let numerator = ref 0. in
      let denominator = ref 0. in
      
      for i = 0 to n_series - 1 do
        let curr_state = Tensor.get history.current i in
        let future = Tensor.get future_state j in
        let joint_prob = Tensor.get state_probs i j in
        
        numerator := !numerator +. 
          (curr_state *. future *. joint_prob);
        denominator := !denominator +. curr_state;
      done;
      
      if !denominator > 0. then
        Tensor.set cond_prob j (!numerator /. !denominator)
    done;
    cond_prob
end

(* State space handling *)
module StateSpace = struct
  type state = int
  type state_space = {
    n_states: int;
    n_series: int;
  }

  (* Create state space configuration *)
  let create ~n_states ~n_series = {
    n_states;
    n_series;
  }

  (* Convert continuous returns to discrete states *)
  let discretize_returns returns n_states =
    let n_samples = Tensor.size returns 0 in
    let n_series = Tensor.size returns 1 in
    let result = Tensor.zeros [|n_samples; n_series|] in
    
    for series = 0 to n_series - 1 do
      let series_data = Tensor.select returns 1 series in
      let sorted, _ = Tensor.sort series_data 0 in
      let quantiles = Array.init (n_states + 1) (fun i ->
        let idx = float_of_int i *. float_of_int n_samples 
          /. float_of_int n_states in
        Tensor.get sorted (int_of_float (floor idx))
      ) in
      
      for sample = 0 to n_samples - 1 do
        let value = Tensor.get series_data sample in
        let state = ref 0 in
        while !state < n_states && value > quantiles.(!state + 1) do
          incr state
        done;
        Tensor.set result sample series (float_of_int !state)
      done
    done;
    result
end

(* Joint probability distribution *)
module JointDistribution = struct
  type distribution_params = {
    eps: float;                 (* Numerical stability parameter *)
    min_prob: float;           (* Minimum probability threshold *)
    regularization: float;     (* Smoothing parameter *)
    check_validity: bool;      (* Enable validity checks *)
  }

  (* Numerical utilities *)
  module NumericUtils = struct
    let safe_log x eps =
      if x <= eps then log eps else log x

    let safe_div num denom eps =
      if abs_float denom < eps then
        if abs_float num < eps then 0.
        else if num > 0. then Float.max_float
        else Float.min_float
      else num /. denom

    let normalize_probs tensor eps =
      let sum = Tensor.sum tensor in
      if sum < eps then
        Tensor.div_scalar tensor eps
      else
        Tensor.div tensor sum

    let smooth_probs tensor alpha =
      let n = Tensor.size tensor 0 in
      let uniform = Tensor.div_scalar 
        (Tensor.ones_like tensor) 
        (float_of_int n) in
      Tensor.add
        (Tensor.mul_scalar tensor (1. -. alpha))
        (Tensor.mul_scalar uniform alpha)
  end

  (* Validation *)
  module Validation = struct
    let check_probability_constraints tensor params =
      let valid = ref true in
      let n = Tensor.size tensor 0 in
      
      for i = 0 to n - 1 do
        let prob = Tensor.get tensor i in
        if prob < 0. || prob < params.min_prob then
          valid := false
      done;
      
      let sum = Tensor.sum tensor in
      if abs_float (sum -. 1.) > params.eps then
        valid := false;
      
      !valid

    let check_markov_property trans_probs params =
      let n = Tensor.size trans_probs 0 in
      let valid = ref true in
      
      for i = 0 to n - 1 do
        let row_sum = Tensor.sum (Tensor.select trans_probs i 0) in
        if abs_float (row_sum -. 1.) > params.eps then
          valid := false
      done;
      
      !valid
  end

  (* Core distribution computations *)
  let compute_joint states params =
    let n_samples = Tensor.size states 0 in
    let n_series = Tensor.size states 1 in
    let n_states = int_of_float (Tensor.max states) + 1 in
    
    let joint = Tensor.zeros 
      [|n_states; n_states; n_series; n_series|] in
    
    for t = 0 to n_samples - 2 do
      for i = 0 to n_series - 1 do
        for j = 0 to n_series - 1 do
          let state_i = int_of_float (Tensor.get states t i) in
          let state_j = int_of_float (Tensor.get states (t+1) j) in
          let curr_count = Tensor.get joint state_i state_j i j in
          Tensor.set joint state_i state_j i j (curr_count +. 1.)
        done
      done
    done;
    
    let joint = NumericUtils.smooth_probs joint params.regularization in
    let joint = NumericUtils.normalize_probs joint params.eps in
    
    if params.check_validity then
      if not (Validation.check_probability_constraints joint params) then
        raise (Invalid_argument "Invalid joint distribution");
    
    joint

  let compute_conditional joint params =
    let shape = Tensor.size joint in
    let n_states = shape.(0) in
    let n_series = shape.(2) in
    let conditional = Tensor.zeros_like joint in
    
    for i = 0 to n_series - 1 do
      for j = 0 to n_series - 1 do
        for s1 = 0 to n_states - 1 do
          let marginal = ref 0. in
          for s2 = 0 to n_states - 1 do
            marginal := !marginal +. Tensor.get joint s1 s2 i j
          done;
          
          for s2 = 0 to n_states - 1 do
            let joint_prob = Tensor.get joint s1 s2 i j in
            let cond_prob = NumericUtils.safe_div 
              joint_prob !marginal params.eps in
            Tensor.set conditional s1 s2 i j cond_prob
          done
        done
      done
    done;
    
    if params.check_validity then
      if not (Validation.check_markov_property conditional params) then
        raise (Invalid_argument "Invalid conditional distribution");
    
    conditional
end

(* MTD model estimation *)
module MTDEstimation = struct
  type estimation_params = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    lambda_constraint_weight: float;
  }

  (* Transition matrix estimation *)
  module TransitionEstimation = struct
    (* Compute empirical transition counts *)
    let compute_counts states n_states =
      let n_samples = Tensor.size states 0 in
      let n_series = Tensor.size states 1 in
      let counts = Array.make_matrix n_series n_series 
        (Tensor.zeros [|n_states; n_states|]) in
      
      for t = 0 to n_samples - 2 do
        for i = 0 to n_series - 1 do
          for j = 0 to n_series - 1 do
            let state_i = int_of_float (Tensor.get states t i) in
            let state_j = int_of_float (Tensor.get states (t+1) j) in
            let curr_count = Tensor.get counts.(i).(j) state_i state_j in
            Tensor.set counts.(i).(j) state_i state_j (curr_count +. 1.)
          done
        done
      done;
      counts

    (* Normalize transition counts to probabilities *)
    let normalize_counts counts =
      let n_series = Array.length counts in
      let probs = Array.make_matrix n_series n_series 
        (Tensor.zeros_like counts.(0).(0)) in
      
      for i = 0 to n_series - 1 do
        for j = 0 to n_series - 1 do
          let row_sums = Tensor.sum counts.(i).(j) ~dim:[1] ~keepdim:true in
          probs.(i).(j) <- Tensor.div counts.(i).(j) row_sums;
          
          (* Add smoothing for numerical stability *)
          let eps = 1e-10 in
          for s1 = 0 to Tensor.size counts.(0).(0) 0 - 1 do
            for s2 = 0 to Tensor.size counts.(0).(0) 1 - 1 do
              let p = Tensor.get probs.(i).(j) s1 s2 in
              if p < eps then
                Tensor.set probs.(i).(j) s1 s2 eps
            done
          done;
          
          (* Renormalize *)
          let row_sums = Tensor.sum probs.(i).(j) ~dim:[1] ~keepdim:true in
          probs.(i).(j) <- Tensor.div probs.(i).(j) row_sums
        done
      done;
      probs
  end

  (* Lambda parameter optimization *)
  module LambdaOptimization = struct
    type optimization_state = {
      lambda: Tensor.t;
      grad_history: Tensor.t list;
      best_lambda: Tensor.t;
      best_likelihood: float;
    }

    (* Compute log-likelihood *)
    let compute_likelihood states trans_probs lambda =
      let n_samples = Tensor.size states 0 in
      let n_series = Tensor.size states 1 in
      let log_lik = ref 0. in
      
      for t = 0 to n_samples - 2 do
        for j = 0 to n_series - 1 do
          let next_state = int_of_float (Tensor.get states (t+1) j) in
          let mut_prob = ref 0. in
          
          for i = 0 to n_series - 1 do
            let curr_state = int_of_float (Tensor.get states t i) in
            let trans_prob = Tensor.get trans_probs.(i).(j) curr_state next_state in
            let weight = Tensor.get lambda i j in
            mut_prob := !mut_prob +. (weight *. trans_prob)
          done;
          
          if !mut_prob > 0. then
            log_lik := !log_lik +. log !mut_prob
        done
      done;
      !log_lik

    (* Compute gradient with constraints *)
    let compute_gradient states trans_probs lambda constraint_weight =
      let n_samples = Tensor.size states 0 in
      let n_series = Tensor.size states 1 in
      let grad = Tensor.zeros_like lambda in
      
      (* Likelihood gradient *)
      for t = 0 to n_samples - 2 do
        for j = 0 to n_series - 1 do
          let next_state = int_of_float (Tensor.get states (t+1) j) in
          let mut_prob = ref 0. in
          let mut_grads = Array.make n_series 0. in
          
          for i = 0 to n_series - 1 do
            let curr_state = int_of_float (Tensor.get states t i) in
            let trans_prob = Tensor.get trans_probs.(i).(j) curr_state next_state in
            let weight = Tensor.get lambda i j in
            mut_prob := !mut_prob +. (weight *. trans_prob);
            mut_grads.(i) <- trans_prob
          done;
          
          if !mut_prob > 0. then
            for i = 0 to n_series - 1 do
              let curr_grad = Tensor.get grad i j in
              let new_grad = curr_grad +. (mut_grads.(i) /. !mut_prob) in
              Tensor.set grad i j new_grad
            done
        done
      done;
      
      (* Constraint gradient *)
      let row_sums = Tensor.sum lambda ~dim:[1] ~keepdim:true in
      let constraint_grad = Tensor.sub row_sums 
        (Tensor.ones_like row_sums) in
      let weighted_constraint = Tensor.mul_scalar constraint_grad 
        constraint_weight in
      
      Tensor.add_ grad weighted_constraint;
      grad

    (* Project onto simplex *)
    let project_simplex lambda =
      let n = Tensor.size lambda 0 in
      let result = Tensor.copy lambda in
      
      for i = 0 to n - 1 do
        let row = Tensor.select result i 0 in
        let sorted, _ = Tensor.sort row 0 ~descending:true in
        let cumsum = Tensor.cumsum sorted ~dim:0 in
        
        let rho = ref (n - 1) in
        for j = 0 to n - 2 do
          let sum_val = Tensor.get cumsum j in
          let val_j = Tensor.get sorted j in
          if val_j > (sum_val -. 1.) /. float_of_int (j + 1) then
            rho := j
        done;
        
        let tau = (Tensor.get cumsum !rho -. 1.) /. 
          float_of_int (!rho + 1) in
          
        for j = 0 to n - 1 do
          let val_j = Tensor.get row j in
          let new_val = max 0. (val_j -. tau) in
          Tensor.set row j new_val
        done
      done;
      result

    (* Optimization procedure *)
    let optimize params states trans_probs init_state =
      let state = ref init_state in
      let converged = ref false in
      let iter = ref 0 in
      
      while not !converged && !iter < params.max_iter do
        (* Compute gradient *)
        let grad = compute_gradient states trans_probs 
          !state.lambda params.lambda_constraint_weight in
          
        (* Update lambda *)
        let new_lambda = Tensor.add !state.lambda 
          (Tensor.mul_scalar grad params.learning_rate) in
        let projected_lambda = project_simplex new_lambda in
        
        (* Compute new likelihood *)
        let new_likelihood = compute_likelihood states trans_probs 
          projected_lambda in
          
        (* Update state *)
        state := { !state with
          lambda = projected_lambda;
          grad_history = grad :: !state.grad_history;
          best_lambda = if new_likelihood > !state.best_likelihood then
            projected_lambda else !state.best_lambda;
          best_likelihood = max new_likelihood !state.best_likelihood;
        };
        
        (* Check convergence *)
        let grad_norm = Tensor.norm grad in
        converged := grad_norm < params.tolerance;
        incr iter
      done;
      !state.best_lambda
  end

  (* Main estimation procedure *)
  let estimate params states =
    let n_samples = Tensor.size states 0 in
    let n_series = Tensor.size states 1 in
    let n_states = int_of_float (Tensor.max states) + 1 in
    
    (* Compute transition probabilities *)
    let counts = TransitionEstimation.compute_counts 
      states n_states in
    let trans_probs = TransitionEstimation.normalize_counts counts in
    
    (* Initialize and optimize lambda parameters *)
    let init_state = {
      LambdaOptimization.
      lambda = Tensor.uniform [|n_series; n_series|] ~low:0. ~high:1.;
      grad_history = [];
      best_lambda = Tensor.zeros [|n_series; n_series|];
      best_likelihood = neg_infinity;
    } in
    
    let lambda = LambdaOptimization.optimize params states 
      trans_probs init_state in
      
    trans_probs, lambda
end

(* Network construction *)
module NetworkConstruction = struct
  type network_params = {
    min_weight: float;
    remove_self_loops: bool;
    weight_normalization: [`None | `Row | `Column | `Global];
  }

  (* Matrix operations *)
  module MatrixOps = struct
    (* Normalize matrix *)
    let normalize matrix scheme =
      let result = Tensor.copy matrix in
      match scheme with
      | `None -> result
      | `Row ->
          let row_sums = Tensor.sum result ~dim:[1] ~keepdim:true in
          Tensor.div result row_sums
      | `Column ->
          let col_sums = Tensor.sum result ~dim:[0] ~keepdim:true in
          Tensor.div result col_sums
      | `Global ->
          let total = Tensor.sum result in
          Tensor.div_scalar result total

    (* Remove self loops *)
    let remove_self_loops matrix =
      let n = Tensor.size matrix 0 in
      let result = Tensor.copy matrix in
      for i = 0 to n - 1 do
        Tensor.set result i i 0.
      done;
      result

    (* Apply threshold *)
    let apply_threshold matrix threshold =
      let mask = Tensor.gt matrix threshold in
      Tensor.mul matrix mask

    (* Convert to adjacency *)
    let to_adjacency weights threshold =
      let mask = Tensor.gt weights threshold in
      Tensor.to_float mask
  end

  (* Network properties computation *)
  module Properties = struct
    (* Compute degree distributions *)
    let compute_degree_dist adj_matrix =
      let n = Tensor.size adj_matrix 0 in
      let in_degrees = Tensor.sum adj_matrix ~dim:[0] in
      let out_degrees = Tensor.sum adj_matrix ~dim:[1] in
      
      let max_degree = int_of_float (max 
        (Tensor.max in_degrees)
        (Tensor.max out_degrees)) in
      let in_dist = Array.make (max_degree + 1) 0 in
      let out_dist = Array.make (max_degree + 1) 0 in
      
      for i = 0 to n - 1 do
        let in_deg = int_of_float (Tensor.get in_degrees i) in
        let out_deg = int_of_float (Tensor.get out_degrees i) in
        in_dist.(in_deg) <- in_dist.(in_deg) + 1;
        out_dist.(out_deg) <- out_dist.(out_deg) + 1
      done;
      
      in_dist, out_dist

    (* Compute weight distribution *)
    let compute_weight_dist weights bins =
      let flat_weights = Tensor.flatten weights in
      let n = Tensor.size flat_weights 0 in
      let min_w = Tensor.min flat_weights in
      let max_w = Tensor.max flat_weights in
      let step = (max_w -. min_w) /. float_of_int bins in
      
      let dist = Array.make bins 0 in
      for i = 0 to n - 1 do
        let w = Tensor.get flat_weights i in
        if w > 0. then
          let bin = int_of_float ((w -. min_w) /. step) in
          let bin = min (bins - 1) bin in
          dist.(bin) <- dist.(bin) + 1
      done;
      dist

    (* Compute clustering coefficients *)
    let compute_clustering adj_matrix =
      let n = Tensor.size adj_matrix 0 in
      let coeffs = Tensor.zeros [|n|] in
      
      for i = 0 to n - 1 do
        let neighbors = ref [] in
        for j = 0 to n - 1 do
          if Tensor.get adj_matrix i j > 0. then
            neighbors := j :: !neighbors
        done;
        
        let k = List.length !neighbors in
        if k > 1 then begin
          let triangles = ref 0 in
          List.iter (fun j ->
            List.iter (fun k ->
              if j <> k && 
                 Tensor.get adj_matrix j k > 0. then
                incr triangles
            ) !neighbors
          ) !neighbors;
          
          let possible = k * (k - 1) in
          Tensor.set coeffs i 
            (float_of_int !triangles /. float_of_int possible)
        end
      done;
      coeffs

    (* Compute number of components *)
    let compute_components network =
      let n = Array.length network.Types.nodes in
      let visited = Array.make n false in
      let component_count = ref 0 in
      
      let rec visit node =
        if not visited.(node) then begin
          visited.(node) <- true;
          for j = 0 to n - 1 do
            if Tensor.get network.Types.adj_matrix node j > 0. then
              visit j
          done
        end in
      
      for i = 0 to n - 1 do
        if not visited.(i) then begin
          incr component_count;
          visit i
        end
      done;
      
      !component_count
  end

  (* Network Construction Module continuation *)
  let create_from_mtd model params =
    let weights = model.lambda_weights in
    
    (* Apply construction steps *)
    let weights = 
      if params.remove_self_loops then
        MatrixOps.remove_self_loops weights
      else weights in
        
    let weights = 
      MatrixOps.normalize weights params.weight_normalization in
        
    let weights = 
      MatrixOps.apply_threshold weights params.min_weight in
        
    let adj_matrix = 
      MatrixOps.to_adjacency weights params.min_weight in
        
    (* Compute network properties *)
    let in_dist, out_dist = 
      Properties.compute_degree_dist adj_matrix in
    let weight_dist = 
      Properties.compute_weight_dist weights 10 in
    let clustering = 
      Properties.compute_clustering adj_matrix in
        
    {
      Types.
      nodes = Array.init (Tensor.size weights 0) (fun i -> i);
      adj_matrix;
      weights_matrix = weights;
      properties = {
        in_degree_distribution = in_dist;
        out_degree_distribution = out_dist;
        weight_distribution = weight_dist;
        clustering_coefficients = clustering;
      }
    }
end

(* Global assortativity *)
module GlobalAssortativity = struct
  type correlation_params = {
    normalization: [`Standard | `Weighted];
    omega_handling: [`Global | `Local];
    excess_type: [`Simple | `Weighted];
  }

  (* Excess strength computation *)
  module ExcessStrength = struct
    (* Compute excess strength *)
    let compute_excess weights strength_type =
      let n = Tensor.size weights 0 in
      let strength = match strength_type with
        | `In -> Tensor.sum weights ~dim:[0]
        | `Out -> Tensor.sum weights ~dim:[1] in
      
      let excess = Tensor.zeros [|n; n|] in
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let s_i = Tensor.get strength i in
          let w_ij = Tensor.get weights i j in
          let ex = match strength_type with
            | `In -> s_i -. w_ij
            | `Out -> s_i -. w_ij in
          Tensor.set excess i j ex
        done
      done;
      excess

    (* Compute weighted excess strength *)
    let compute_weighted_excess weights strength_type =
      let n = Tensor.size weights 0 in
      let total_weight = Tensor.sum weights in
      let strength = match strength_type with
        | `In -> 
            let str = Tensor.sum weights ~dim:[0] in
            Tensor.div str total_weight
        | `Out ->
            let str = Tensor.sum weights ~dim:[1] in
            Tensor.div str total_weight in
      
      let excess = Tensor.zeros [|n; n|] in
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let s_i = Tensor.get strength i in
          let w_ij = Tensor.get weights i j /. total_weight in
          let ex = match strength_type with
            | `In -> s_i -. w_ij
            | `Out -> s_i -. w_ij in
          Tensor.set excess i j ex
        done
      done;
      excess
  end

  (* Correlation computation *)
  module Correlation = struct
    (* Compute weighted mean *)
    let weighted_mean values weights =
      let total_weight = Tensor.sum weights in
      let weighted_sum = Tensor.sum (Tensor.mul values weights) in
      Tensor.div_scalar weighted_sum total_weight

    (* Compute weighted standard deviation *)
    let weighted_std values weights mean =
      let total_weight = Tensor.sum weights in
      let sq_dev = Tensor.mul 
        (Tensor.sub values mean) 
        (Tensor.sub values mean) in
      let weighted_sq_dev = Tensor.mul sq_dev weights in
      let variance = Tensor.div_scalar 
        (Tensor.sum weighted_sq_dev) 
        total_weight in
      sqrt variance

    (* Compute weighted correlation *)
    let weighted_correlation ex1 ex2 weights params =
      let n = Tensor.size weights 0 in
      let omega = match params.omega_handling with
        | `Global -> Tensor.sum weights
        | `Local -> 
            let row_sums = Tensor.sum weights ~dim:[1] in
            Tensor.mean row_sums in
      
      (* Compute weighted means *)
      let mean1 = weighted_mean ex1 weights in
      let mean2 = weighted_mean ex2 weights in
      
      (* Compute weighted standard deviations *)
      let std1 = weighted_std ex1 weights mean1 in
      let std2 = weighted_std ex2 weights mean2 in
      
      (* Compute correlation numerator *)
      let dev1 = Tensor.sub ex1 mean1 in
      let dev2 = Tensor.sub ex2 mean2 in
      let cross_dev = Tensor.mul dev1 dev2 in
      let weighted_cross = Tensor.mul cross_dev weights in
      let numerator = Tensor.sum weighted_cross in
      
      (* Apply normalization *)
      match params.normalization with
      | `Standard ->
          numerator /. (std1 *. std2 *. omega)
      | `Weighted ->
          let w_sum = Tensor.sum weights in
          numerator /. (std1 *. std2 *. w_sum)
  end

  (* Compute global assortativity *)
  let compute network params =
    let weights = network.Types.weights_matrix in
    
    (* Compute excess strengths *)
    let ex_in, ex_out = match params.excess_type with
      | `Simple -> 
          ExcessStrength.compute_excess weights `In,
          ExcessStrength.compute_excess weights `Out
      | `Weighted ->
          ExcessStrength.compute_weighted_excess weights `In,
          ExcessStrength.compute_weighted_excess weights `Out in
    
    (* Compute correlations for different modes *)
    let in_in = 
      Correlation.weighted_correlation ex_in ex_in weights params in
    let in_out =
      Correlation.weighted_correlation ex_in ex_out weights params in
    let out_in =
      Correlation.weighted_correlation ex_out ex_in weights params in
    let out_out =
      Correlation.weighted_correlation ex_out ex_out weights params in
    
    {
      Types.
      in_in;
      in_out;
      out_in;
      out_out;
      total = (in_in +. in_out +. out_in +. out_out) /. 4.0;
    }
end

(* Local assortativity measures *)
module LocalAssortativity = struct
  (* Extension for local assortativity *)
  module LocalAssortExtension = struct
    (* Compute normalized local contribution *)
    let compute_local_contribution network node params =
      let n = Array.length network.Types.nodes in
      let weights = network.Types.weights_matrix in
      
      (* Get node's local weights *)
      let local_weights = Tensor.zeros [|n|] in
      for j = 0 to n - 1 do
        if node <> j then
          Tensor.set local_weights j 
            (Tensor.get weights node j)
      done;
      
      (* Compute local excess strengths *)
      let ex_in = GlobalAssortativity.ExcessStrength.compute_excess 
        weights `In in
      let ex_out = GlobalAssortativity.ExcessStrength.compute_excess
        weights `Out in
      
      (* Get node's excess values *)
      let node_ex_in = Tensor.select ex_in node 0 in
      let node_ex_out = Tensor.select ex_out node 0 in
      
      (* Compute local correlation *)
      let local_corr = 
        GlobalAssortativity.Correlation.weighted_correlation
          node_ex_in node_ex_out local_weights params in
      
      (* Normalize by global statistics *)
      let global_assort = GlobalAssortativity.compute network params in
      let normalized = local_corr /. global_assort.total in
      normalized

    (* Compute all local assortativity values *)
    let compute_all network params =
      let n = Array.length network.Types.nodes in
      let result = Tensor.zeros [|n|] in
      
      for i = 0 to n - 1 do
        let local = compute_local_contribution network i params in
        Tensor.set result i local
      done;
      result
  end

(* Enhanced Sabek measure *)
module EnhancedSabek = struct
  type edge_params = {
    normalization: [`Local | `Global];
    weight_handling: [`Raw | `Normalized];
    neighbor_type: [`Direct | `Extended];
  }

  let default_params = {
    normalization = `Local;
    weight_handling = `Normalized;
    neighbor_type = `Extended;
  }

  (* Edge assortativity computation *)
  module EdgeAssortativity = struct
    (* Compute weighted edge statistics *)
    let compute_edge_stats network params =
      let n = Array.length network.Types.nodes in
      let weights = network.Types.weights_matrix in
      
      (* Compute means based on normalization type *)
      let means = match params.normalization with
        | `Local ->
            let row_means = Tensor.mean weights ~dim:[1] ~keepdim:true in
            let col_means = Tensor.mean weights ~dim:[0] ~keepdim:true in
            row_means, col_means
        | `Global ->
            let global_mean = Tensor.mean weights in
            Tensor.full [|n; 1|] global_mean,
            Tensor.full [|1; n|] global_mean in
            
      (* Compute standard deviations *)
      let compute_std values mean =
        let dev = Tensor.sub values mean in
        let sq_dev = Tensor.mul dev dev in
        sqrt (Tensor.mean sq_dev) in
        
      let std_i = compute_std weights (fst means) in
      let std_j = compute_std weights (snd means) in
      
      means, (std_i, std_j)

    (* Compute edge contribution *)
    let compute_edge_contribution weights i j stats params =
      let (means_i, means_j), (std_i, std_j) = stats in
      let w_ij = Tensor.get weights i j in
      
      match params.weight_handling with
      | `Raw -> 
          let mean_i = Tensor.get means_i i 0 in
          let mean_j = Tensor.get means_j 0 j in
          w_ij *. (mean_i -. mean_j) /. (std_i *. std_j)
      | `Normalized ->
          let total = Tensor.sum weights in
          let w_ij = w_ij /. total in
          let mean_i = Tensor.get means_i i 0 /. total in
          let mean_j = Tensor.get means_j 0 j /. total in
          w_ij *. (mean_i -. mean_j) /. (std_i *. std_j)
  end

  (* Neighborhood computation *)
  module Neighborhood = struct
    (* Get extended neighborhood *)
    let get_extended_neighbors network node depth =
      let n = Array.length network.Types.nodes in
      let visited = Array.make n false in
      let neighbors = ref [] in
      let weights = network.Types.weights_matrix in
      
      let rec visit curr_node curr_depth =
        if curr_depth <= depth && not visited.(curr_node) then begin
          visited.(curr_node) <- true;
          if curr_node <> node then
            neighbors := curr_node :: !neighbors;
          
          for j = 0 to n - 1 do
            if Tensor.get weights curr_node j > 0. then
              visit j (curr_depth + 1)
          done
        end in
      
      visit node 0;
      Array.of_list !neighbors

    (* Get direct neighbors *)
    let get_direct_neighbors network node =
      let n = Array.length network.Types.nodes in
      let neighbors = ref [] in
      let weights = network.Types.weights_matrix in
      
      for j = 0 to n - 1 do
        if node <> j && Tensor.get weights node j > 0. then
          neighbors := j :: !neighbors
      done;
      Array.of_list !neighbors
  end

  (* Compute node assortativity *)
  let compute_node_assortativity network node params =
    let weights = network.Types.weights_matrix in
    let stats = EdgeAssortativity.compute_edge_stats network params in
    
    let neighbors = match params.neighbor_type with
      | `Direct -> 
          Neighborhood.get_direct_neighbors network node
      | `Extended ->
          Neighborhood.get_extended_neighbors network node 2 in
          
    let contributions = ref 0. in
    Array.iter (fun j ->
      let contrib = EdgeAssortativity.compute_edge_contribution
        weights node j stats params in
      contributions := !contributions +. contrib
    ) neighbors;
    
    !contributions
end

(* Enhanced Peel measure *)
module EnhancedPeel = struct
  type multiscale_params = {
    n_scales: int;
    alpha_range: float * float;
    integration_method: [`Uniform | `Weighted];
    scale_weights: float array option;
  }

  (* PageRank computation *)
  module PageRank = struct
    let compute network alpha target =
      let n = Array.length network.Types.nodes in
      let weights = network.Types.weights_matrix in
      
      (* Normalize transition probabilities *)
      let trans = Tensor.zeros [|n; n|] in
      for i = 0 to n - 1 do
        let row_sum = Tensor.sum (Tensor.select weights i 0) in
        if row_sum > 0. then
          for j = 0 to n - 1 do
            let w_ij = Tensor.get weights i j in
            Tensor.set trans i j (w_ij /. row_sum)
          done
      done;
      
      (* Power iteration *)
      let pr = Tensor.ones [|n|] in
      Tensor.div_scalar_ pr (float_of_int n);
      let teleport = Tensor.zeros [|n|] in
      Tensor.set teleport target 1.;
      
      let converged = ref false in
      let max_iter = 100 in
      let iter = ref 0 in
      let tol = 1e-6 in
      
      while not !converged && !iter < max_iter do
        let old_pr = Tensor.copy pr in
        
        (* Random walk *)
        let walk = Tensor.matmul 
          (Tensor.unsqueeze pr 0) trans in
        let walk = Tensor.squeeze walk 0 in
        
        (* Combine with teleportation *)
        pr <- Tensor.add
          (Tensor.mul_scalar walk alpha)
          (Tensor.mul_scalar teleport (1. -. alpha));
          
        (* Check convergence *)
        let diff = Tensor.sub pr old_pr in
        let max_diff = Tensor.max (Tensor.abs diff) in
        converged := max_diff < tol;
        incr iter
      done;
      pr

    (* Compute multiscale PageRank *)
    let compute_multiscale network params target =
      let alphas = match params.integration_method with
        | `Uniform ->
            let min_alpha, max_alpha = params.alpha_range in
            let step = (max_alpha -. min_alpha) /. 
              float_of_int (params.n_scales - 1) in
            Array.init params.n_scales (fun i ->
              min_alpha +. float_of_int i *. step)
        | `Weighted ->
            match params.scale_weights with
            | Some weights -> 
                let min_alpha, max_alpha = params.alpha_range in
                Array.init params.n_scales (fun i ->
                  min_alpha +. weights.(i) *. (max_alpha -. min_alpha))
            | None -> 
                let min_alpha, max_alpha = params.alpha_range in
                let step = (max_alpha -. min_alpha) /. 
                  float_of_int (params.n_scales - 1) in
                Array.init params.n_scales (fun i ->
                  min_alpha +. float_of_int i *. step) in
      
      let n = Array.length network.Types.nodes in
      let result = Tensor.zeros [|n|] in
      
      Array.iteri (fun i alpha ->
        let pr = compute network alpha target in
        let weight = match params.integration_method with
          | `Uniform -> 1. /. float_of_int params.n_scales
          | `Weighted ->
              match params.scale_weights with
              | Some weights -> weights.(i)
              | None -> 1. /. float_of_int params.n_scales in
        Tensor.add_ result (Tensor.mul_scalar pr weight)
      ) alphas;
      
      result
  end

  (* Local assortativity computation *)
  let compute_local network params =
    let n = Array.length network.Types.nodes in
    let result = Tensor.zeros [|n|] in
    
    for l = 0 to n - 1 do
      let pr_dist = PageRank.compute_multiscale network params l in
      let local_assort = ref 0. in
      
      let ex_in = GlobalAssortativity.ExcessStrength.compute_excess 
        network.Types.weights_matrix `In in
      let ex_out = GlobalAssortativity.ExcessStrength.compute_excess
        network.Types.weights_matrix `Out in
      
      for i = 0 to n - 1 do
        let pr_i = Tensor.get pr_dist i in
        for j = 0 to n - 1 do
          let w_ij = Tensor.get network.Types.weights_matrix i j in
          if w_ij > 0. then
            let es_in = Tensor.get ex_in i j in
            let es_out = Tensor.get ex_out i j in
            local_assort := !local_assort +. 
              (pr_i *. w_ij *. es_in *. es_out)
        done
      done;
      
      Tensor.set result l !local_assort
    done;
    
    result
end

(* Portfolio optimization *)
module Portfolio = struct
  type portfolio_weights = Tensor.t
  
  type optimization_result = {
    weights: portfolio_weights;
    objective_value: float;
    network_metrics: network_metrics;
    convergence: convergence_info;
  }
  and network_metrics = {
    assortativity: float;
    centrality: float;
    risk_contribution: float;
  }
  and convergence_info = {
    iterations: int;
    final_grad_norm: float;
    converged: bool;
  }

  (* Utility functions *)
  module Utils = struct
    let portfolio_return weights returns =
      Tensor.dot weights returns
      
    let portfolio_variance weights covariance =
      let temp = Tensor.matmul covariance (Tensor.unsqueeze weights 1) in
      Tensor.dot weights (Tensor.squeeze temp 1)
      
    let portfolio_sharpe weights returns covariance rf =
      let ret = portfolio_return weights returns in
      let var = portfolio_variance weights covariance in
      (ret -. rf) /. (sqrt var)
  end

  (* Portfolio constraints *)
  module Constraints = struct
    type constraints = {
      min_weight: float;
      max_weight: float;
      sum_weights: float;
      long_only: bool;
    }

    let default = {
      min_weight = 0.;
      max_weight = 1.;
      sum_weights = 1.;
      long_only = true;
    }

    let apply_constraints constraints weights =
      let n = Tensor.size weights 0 in
      let result = Tensor.copy weights in
      
      if constraints.long_only then
        for i = 0 to n - 1 do
          if Tensor.get result i < 0. then
            Tensor.set result i 0.
        done;

      for i = 0 to n - 1 do
        let w = Tensor.get result i in
        let w' = max constraints.min_weight 
          (min w constraints.max_weight) in
        Tensor.set result i w'
      done;

      let sum_weights = Tensor.sum result in
      if sum_weights > 0. then
        Tensor.mul_scalar_ result 
          (constraints.sum_weights /. sum_weights)
      else
        Tensor.div_scalar_ 
          (Tensor.ones [|n|]) 
          (float_of_int n);
      
      result
  end
end

(* Enhanced portfolio optimization with network measures *)
module PortfolioOptimizer = struct
  type optimization_params = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    risk_aversion: float;
    risk_free_rate: float;
    network_weight: float;
  }

  (* Gradient computation *)
  module Gradients = struct
    (* Compute return gradient *)
    let compute_return_gradient returns weights =
      returns

    (* Compute risk gradient *)
    let compute_risk_gradient covariance weights risk_aversion =
      let risk_contribution = Tensor.matmul covariance 
        (Tensor.unsqueeze weights 1) in
      let grad = Tensor.squeeze risk_contribution 1 in
      Tensor.mul_scalar grad risk_aversion

    (* Compute network gradient *)
    let compute_network_gradient network weights network_weight =
      let n = Array.length network.Types.nodes in
      let assort_dist = 
        LocalAssortativity.LocalAssortExtension.compute_all 
          network GlobalAssortativity.default_params in
      Tensor.mul_scalar assort_dist network_weight

    (* Compute total gradient *)
    let compute_gradient params returns covariance network weights =
      let return_grad = compute_return_gradient returns weights in
      let risk_grad = compute_risk_gradient covariance weights 
        params.risk_aversion in
      let network_grad = compute_network_gradient network weights 
        params.network_weight in
      
      Tensor.sub
        (Tensor.sub return_grad risk_grad)
        network_grad
  end

  (* Objective function computation *)
  module Objective = struct
    (* Compute portfolio return *)
    let compute_return returns weights =
      Portfolio.Utils.portfolio_return weights returns

    (* Compute portfolio risk *)
    let compute_risk covariance weights risk_aversion =
      let var = Portfolio.Utils.portfolio_variance weights covariance in
      risk_aversion *. var /. 2.

    (* Compute network contribution *)
    let compute_network_contribution network weights network_weight =
      let assort = LocalAssortativity.LocalAssortExtension.compute_all 
        network GlobalAssortativity.default_params in
      network_weight *. Tensor.dot weights assort

    (* Compute total objective *)
    let compute params returns covariance network weights =
      let ret = compute_return returns weights in
      let risk = compute_risk covariance weights params.risk_aversion in
      let network = compute_network_contribution network weights 
        params.network_weight in
      ret -. risk -. network
  end

  (* Portfolio optimization with network measures *)
  let optimize params returns covariance network initial_weights =
    let n = Tensor.size returns 0 in
    let weights = ref (match initial_weights with
      | Some w -> Tensor.copy w
      | None -> Tensor.div_scalar (Tensor.ones [|n|]) 
        (float_of_int n)) in
    let best_weights = ref !weights in
    let best_objective = ref neg_infinity in
    
    let iter = ref 0 in
    let converged = ref false in
    let final_grad_norm = ref 0. in
    
    while not !converged && !iter < params.max_iter do
      (* Compute gradient *)
      let grad = Gradients.compute_gradient params returns 
        covariance network !weights in
      
      (* Take gradient step *)
      let new_weights = Tensor.add !weights 
        (Tensor.mul_scalar grad params.learning_rate) in
      
      (* Project onto constraints *)
      let projected_weights = 
        Portfolio.Constraints.apply_constraints 
          Portfolio.Constraints.default new_weights in
      
      (* Compute objective *)
      let obj = Objective.compute params returns covariance 
        network projected_weights in
      
      (* Update best solution *)
      if obj > !best_objective then begin
        best_objective := obj;
        best_weights := Tensor.copy projected_weights
      end;
      
      (* Update weights *)
      weights := projected_weights;
      
      (* Check convergence *)
      final_grad_norm := Tensor.norm grad;
      converged := !final_grad_norm < params.tolerance;
      incr iter
    done;
    
    (* Compute final metrics *)
    let centrality = NetworkAnalysis.Centrality.compute_eigenvector_centrality
      network.Types.adj_matrix in
    let assort = LocalAssortativity.LocalAssortExtension.compute_all 
      network GlobalAssortativity.default_params in
    let risk_contrib = Portfolio.Utils.portfolio_variance 
      !best_weights covariance in
    
    {
      Portfolio.
      weights = !best_weights;
      objective_value = !best_objective;
      network_metrics = {
        assortativity = Tensor.dot !best_weights assort;
        centrality = Tensor.dot !best_weights centrality;
        risk_contribution = risk_contrib;
      };
      convergence = {
        iterations = !iter;
        final_grad_norm = !final_grad_norm;
        converged = !converged;
      };
    }
end

(* Integration *)
module NetworkPortfolioIntegration = struct
  type optimization_config = {
    network_params: MultiscaleAnalysis.scale_params;
    portfolio_params: PortfolioOptimizer.optimization_params;
    estimation_params: MTDEstimation.estimation_params;
  }

  type optimization_result = {
    portfolio: Portfolio.optimization_result;
    network_analysis: network_analysis;
    validation: validation_info;
  }
  and network_analysis = {
    global_assortativity: Types.assortativity_measures;
    local_assortativity: Tensor.t;
    multiscale_measures: Tensor.t;
  }
  and validation_info = {
    warnings: string list;
    diagnostics: diagnostic_info;
  }
  and diagnostic_info = {
    numerical_stability: bool;
    constraint_violation: bool;
    convergence_quality: float;
  }

  (* Validation utilities *)
  module Validation = struct
    let check_inputs returns covariance network =
      let warnings = ref [] in
      
      (* Check dimensions *)
      let n_assets = Tensor.size returns 0 in
      if Tensor.size covariance 0 <> n_assets || 
         Tensor.size covariance 1 <> n_assets then
        warnings := "Dimension mismatch in inputs" :: !warnings;
      
      (* Check for NaN/Inf *)
      let check_tensor tensor name =
        if Tensor.isnan tensor |> Tensor.any then
          warnings := ("NaN values in " ^ name) :: !warnings;
        if Tensor.isinf tensor |> Tensor.any then
          warnings := ("Infinite values in " ^ name) :: !warnings
      in
      
      check_tensor returns "returns";
      check_tensor covariance "covariance";
      check_tensor network.Types.weights_matrix "network weights";
      
      !warnings

    let check_numerical_stability weights obj grad_norm =
      let stable = ref true in
      
      if abs_float obj > 1e6 then
        stable := false;
      
      if grad_norm < 1e-10 then
        stable := false;
      
      let weight_range = Tensor.max weights -. Tensor.min weights in
      if weight_range < 1e-6 then
        stable := false;
      
      !stable

    let compute_convergence_quality iter max_iter grad_norm =
      let iter_quality = 1. -. (float_of_int iter /. 
        float_of_int max_iter) in
      let grad_quality = exp (-. grad_norm) in
      (iter_quality +. grad_quality) /. 2.
  end

  (* Analysis computation *)
  module Analysis = struct
    let compute_network_measures network config =
      (* Global assortativity *)
      let global_assort = GlobalAssortativity.compute network
        GlobalAssortativity.default_params in
      
      (* Local assortativity *)
      let local_assort = 
        LocalAssortativity.LocalAssortExtension.compute_all
          network GlobalAssortativity.default_params in
      
      (* Multiscale measures *)
      let multiscale = MultiscaleAnalysis.Measures.compute_multiscale_assortativity
        network 
        (Array.init config.network_params.n_scales (fun i ->
          float_of_int (i + 1)))
        (Array.make config.network_params.n_scales 
          (1. /. float_of_int config.network_params.n_scales)) in
      
      {
        global_assortativity = global_assort;
        local_assortativity = local_assort;
        multiscale_measures = multiscale;
      }
  end

  (* Main optimization procedure *)
  let optimize config returns covariance initial_network =
    (* Initial validation *)
    let warnings = Validation.check_inputs returns covariance 
      initial_network in
    
    (* Estimate MTD model *)
    let states = StateSpace.discretize_returns returns 
      config.estimation_params.max_iter in
    let trans_probs, lambda = MTDEstimation.estimate 
      config.estimation_params states in
    
    (* Create network from MTD model *)
    let network = NetworkConstruction.create_from_mtd
      { lambda_weights = lambda }
      { min_weight = 0.01; 
        remove_self_loops = true;
        weight_normalization = `Global } in
    
    (* Compute network measures *)
    let network_analysis = Analysis.compute_network_measures 
      network config in
    
    (* Optimize portfolio *)
    let portfolio_result = PortfolioOptimizer.optimize
      config.portfolio_params returns covariance network None in
    
    (* Validate results *)
    let numerically_stable = Validation.check_numerical_stability
      portfolio_result.weights
      portfolio_result.objective_value
      portfolio_result.convergence.final_grad_norm in
    
    let constraint_violated = 
      abs_float (Tensor.sum portfolio_result.weights -. 1.) > 1e-6 in
    
    let convergence_quality = Validation.compute_convergence_quality
      portfolio_result.convergence.iterations
      config.portfolio_params.max_iter
      portfolio_result.convergence.final_grad_norm in
    
    {
      portfolio = portfolio_result;
      network_analysis;
      validation = {
        warnings;
        diagnostics = {
          numerical_stability = numerically_stable;
          constraint_violation = constraint_violated;
          convergence_quality;
        }
      }
    }
end