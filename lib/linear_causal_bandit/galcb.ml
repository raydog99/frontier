open Graph

module ActionSet = Set.Make(NodeSet)

type observation = {
  values: Tensor.t;
  intervention: NodeSet.t;
  timestamp: int;
  reward: float;
}

type config = {
  alpha: float;
  min_exploration_rounds: int;
  phases: int;
  epsilon: float;
}

type state = {
  graph: Graph.t;
  model: Linear_sem.t;
  t: int;
  observations: observation list;
  exploration_rounds: int;
  eta: float;
  config: config;
  current_phase: int;
  candidate_actions: ActionSet.t;
  cumulative_regret: float;
}

let default_config = {
  alpha = 1.0;
  min_exploration_rounds = 100;
  phases = 0;
  epsilon = 0.01;
}

module StructureLearning = struct
  type ancestor_cache = (int, NodeSet.t) Hashtbl.t

  let estimate_descendants state observations threshold =
    let dim = state.model.dim in
    let descendants = Array.make dim NodeSet.empty in
    
    for i = 0 to dim - 1 do
      let null_mean, int_means = 
        Linear_sem.estimate_means observations (NodeSet.singleton i)
      in
      
      for j = 0 to dim - 1 do
        let int_mean = NodeMap.find i int_means in
        let diff = Tensor.(abs (int_mean - null_mean)) in
        if Tensor.float_value diff > threshold then
          descendants.(i) <- NodeSet.add j descendants.(i)
      done
    done;
    descendants
    
  let estimate_ancestors state desc_sets =
    let dim = state.model.dim in
    let ancestors = Array.make dim NodeSet.empty in
    let cache = Hashtbl.create dim in
    
    let rec get_ancestors node =
      match Hashtbl.find_opt cache node with
      | Some ans -> ans
      | None ->
          let pa = NodeSet.elements desc_sets.(node) in
          let ans = List.fold_left (fun acc p ->
            if p <> node then
              let p_ans = get_ancestors p in
              NodeSet.union acc (NodeSet.add p p_ans)
            else acc
          ) NodeSet.empty pa in
          Hashtbl.add cache node ans;
          ans
    in
    
    for i = 0 to dim - 1 do
      ancestors.(i) <- get_ancestors i
    done;
    (ancestors, cache)

  let validate_dag ancestors dim =
    let visited = Array.make dim false in
    let rec_stack = Array.make dim false in
    
    let rec has_cycle node =
      if not visited.(node) then begin
        visited.(node) <- true;
        rec_stack.(node) <- true;
        
        let cycle_found = ref false in
        NodeSet.iter (fun neighbor ->
          if (not visited.(neighbor) && has_cycle neighbor) ||
             rec_stack.(neighbor) then
            cycle_found := true
        ) ancestors.(node);
        
        rec_stack.(node) <- false;
        !cycle_found
      end else
        rec_stack.(node)
    in
    
    let has_any_cycle = ref false in
    for node = 0 to dim - 1 do
      if not visited.(node) && has_cycle node then
        has_any_cycle := true
    done;
    not !has_any_cycle

  let estimate_parents state ancestors observations =
    let dim = state.model.dim in
    let parents = Array.make dim NodeSet.empty in
    
    let estimate_parent_set node ancestor_set =
      if not (NodeSet.is_empty ancestor_set) then begin
        let null_obs = List.filter (fun obs -> 
          NodeSet.is_empty obs.intervention
        ) observations in
        
        let an_list = NodeSet.elements ancestor_set in
        let n = List.length null_obs in
        
        (* Form design matrix X and response y *)
        let X = Tensor.stack (List.map (fun obs ->
          let vals = obs.values in
          Tensor.stack (List.map (fun j -> vals.(.j)) an_list) ~dim:0
        ) null_obs) in
        let y = Tensor.stack (List.map (fun obs -> 
          obs.values.(.node)
        ) null_obs) in
        
        (* LASSO estimation *)
        let lambda = sqrt (2.0 *. log (4.0 *. float dim *. 
                    float (NodeSet.cardinal ancestor_set)) /. float n) in
        
        let beta = Estimation.Lasso.coordinate_descent 
          ~x:X ~y ~lambda ~max_iter:100 in
        
        (* Convert non-zero coefficients to parent set *)
        List.iteri (fun idx j ->
          if abs_float (Tensor.float_value beta.(.idx)) > 1e-6 then
            parents.(node) <- NodeSet.add j parents.(node)
        ) an_list
      end
    in
    
    for i = 0 to dim - 1 do
      estimate_parent_set i ancestors.(i)
    done;
    parents

  let calculate_exploration_rounds dim eta delta =
    let t1 = int_of_float (
      32.0 *. float (dim * dim) /. (eta *. eta) *. 
      log (2.0 *. float (dim * dim) /. delta)
    ) in
    let t2 = int_of_float (
      float dim *. log (float dim) *. log (1.0 /. delta)
    ) in
    (max t1 1, max t2 1)

  let learn state =
    let t_ref = ref 0 in
    let observations = ref [] in
    
    (* Collect initial observations *)
    for _ = 1 to state.exploration_rounds do
      (* Pull null intervention *)
      let obs = {
        values = Linear_sem.simulate state.model NodeSet.empty 
                 (Tensor.randn [state.model.dim]);
        intervention = NodeSet.empty;
        timestamp = !t_ref;
        reward = 0.0;  (* Will be set later *)
      } in
      observations := obs :: !observations;
      incr t_ref;
      
      (* Pull single node interventions *)
      for i = 0 to state.model.dim - 1 do
        let int_set = NodeSet.singleton i in
        let obs = {
          values = Linear_sem.simulate state.model int_set 
                   (Tensor.randn [state.model.dim]);
          intervention = int_set;
          timestamp = !t_ref;
          reward = 0.0;
        } in
        observations := obs :: !observations;
        incr t_ref
      done
    done;
    
    (* Structure learning *)
    let descendants = estimate_descendants state !observations (state.eta /. 2.0) in
    let ancestors, cache = estimate_ancestors state descendants in
    
    (* Validate DAG structure *)
    if not (validate_dag ancestors state.model.dim) then
      failwith "Invalid DAG structure detected";
      
    let parents = estimate_parents state ancestors !observations in
    
    (* Convert to NodeMap *)
    Array.fold_left (fun acc (i, p) -> 
      NodeMap.add i p acc
    ) NodeMap.empty (Array.mapi (fun i p -> (i, p)) parents)
end

module InterventionDesign = struct
  type width_cache = {
    mutable values: float NodeMap.t NodeMap.t;
  }
  
  type mean_estimate = {
    value: float;
    confidence: float;
  }
  
  let create_width_cache () = {
    values = NodeMap.empty
  }
  
  let get_width cache action node =
    match NodeMap.find_opt action cache.values with
    | None -> None
    | Some node_map -> NodeMap.find_opt node node_map
    
  let set_width cache action node width =
    let node_map = 
      match NodeMap.find_opt action cache.values with
      | None -> NodeMap.singleton node width
      | Some m -> NodeMap.add node width m
    in
    cache.values <- NodeMap.add action node_map cache.values

  let estimate_weight_matrices state observations =
    let dim = state.model.dim in
    let parents = state.graph.parents in
    
    (* Initialize matrices *)
    let b = Tensor.zeros [dim; dim] in
    let b_star = Tensor.zeros [dim; dim] in
    
    (* Create gram matrices *)
    let create_gram_matrix obs_filter =
      let obs = List.filter obs_filter observations in
      
      if List.length obs > 0 then
        let x = Tensor.stack (List.map (fun o -> o.values) obs) in
        let xt = Tensor.transpose x ~dim0:0 ~dim1:1 in
        Some (Tensor.matmul xt x)
      else None
    in
    
    (* Estimate for each node *)
    NodeMap.iter (fun i pa_i ->
      let pa_list = NodeSet.elements pa_i in
      let p = List.length pa_list in
      
      (* Observational data *)
      let gram_obs = create_gram_matrix (fun obs -> 
        not (NodeSet.mem i obs.intervention)) in
      
      Option.iter (fun gram ->
        let y = Tensor.stack (List.filter_map (fun obs ->
          if not (NodeSet.mem i obs.intervention) then
            Some obs.values.(.i)
          else None
        ) observations) in
        
        let beta = Estimation.RidgeRegression.estimate 
          ~x:gram ~y ~lambda:1.0 in
        
        List.iteri (fun idx j ->
          Tensor.set b [|i; j|] (Tensor.get beta [|idx|])
        ) pa_list
      ) gram_obs;
      
      (* Interventional data *)
      let gram_int = create_gram_matrix (fun obs ->
        NodeSet.mem i obs.intervention) in
      
      Option.iter (fun gram ->
        let y = Tensor.stack (List.filter_map (fun obs ->
          if NodeSet.mem i obs.intervention then
            Some obs.values.(.i)
          else None
        ) observations) in
        
        let beta = Estimation.RidgeRegression.estimate 
          ~x:gram ~y ~lambda:1.0 in
        
        List.iteri (fun idx j ->
          Tensor.set b_star [|i; j|] (Tensor.get beta [|idx|])
        ) pa_list
      ) gram_int
    ) parents;
    
    (b, b_star)

  let calculate_width state cache parents_depths action node =
    match get_width cache action node with
    | Some w -> w
    | None ->
        let pa_node = Graph.get_parents state.graph node in
        let width =
          if NodeSet.is_empty pa_node then 0.0
          else begin
            (* Calculate recursive width per equation (19) *)
            let pa_widths = NodeSet.fold (fun p acc ->
              acc +. calculate_width state cache parents_depths action p
            ) pa_node 0.0 in
            
            let alpha = sqrt (0.5 *. log (float state.model.dim *. float state.t)) +. 
                       sqrt (float state.graph.max_in_degree) in
            
            pa_widths +. alpha *. (float (NodeSet.cardinal pa_node) +. 
                                 float parents_depths.(node))
          end
        in
        set_width cache action node width;
        width

  let estimate_mean observations node action =
    let relevant_obs = List.filter (fun obs -> 
      NodeSet.equal obs.intervention action
    ) observations in
    
    if List.length relevant_obs = 0 then
      None
    else
      let values = List.map (fun obs -> obs.values.(.node)) relevant_obs in
      let mean = Tensor.(stack values |> Stats.mean |> float_value) in
      let n = float (List.length values) in
      let confidence = sqrt (2.0 *. log (1.0 /. 0.05) /. n) in
      Some { value = mean; confidence }

  let calculate_ucb state node action =
    match estimate_mean state.observations node action with
    | None -> None
    | Some est -> 
        let width = calculate_width state (create_width_cache ()) 
                     (Array.make state.model.dim 0) action node in
        Some (est.value +. est.confidence +. width)

  let should_eliminate state action best_ucb =
    match calculate_ucb state (state.model.dim - 1) action with
    | None -> true
    | Some ucb -> 
        let threshold = 2.0 ** (float (-state.current_phase)) in
        ucb < best_ucb -. threshold

  let refine_actions state =
    let best_ucb = ref neg_infinity in
    
    (* Find best UCB *)
    ActionSet.iter (fun action ->
      match calculate_ucb state (state.model.dim - 1) action with
      | None -> ()
      | Some ucb -> best_ucb := max !best_ucb ucb
    ) state.candidate_actions;
    
    (* Eliminate actions *)
    let new_candidates = ActionSet.filter (fun action ->
      not (should_eliminate state action !best_ucb)
    ) state.candidate_actions in
    
    (* Update state with new candidates *)
    { state with 
      candidate_actions = new_candidates;
      current_phase = state.current_phase + 1 
    }

  let is_width_small state action =
    let cache = create_width_cache () in
    let depths = Array.make state.model.dim 0 in
    let width = calculate_width state cache depths action (state.model.dim - 1) in
    width <= 1.0 /. sqrt (float state.t)

  let select_best_action state =
    let best_action = ref NodeSet.empty in
    let best_ucb = ref neg_infinity in
    
    ActionSet.iter (fun action ->
      match calculate_ucb state (state.model.dim - 1) action with
      | None -> ()
      | Some ucb ->
          if ucb > !best_ucb then begin
            best_ucb := ucb;
            best_action := action
          end
    ) state.candidate_actions;
    
    !best_action

  let select_action state =
    (* Check if we can stop exploration *)
    let all_small_width = ActionSet.for_all (fun action ->
      is_width_small state action
    ) state.candidate_actions in
    
    if all_small_width then
      select_best_action state
    else begin
      (* Find underexplored action *)
      let under_explored = ActionSet.filter (fun action ->
        not (is_width_small state action)
      ) state.candidate_actions in
      
      (* Pick random underexplored action *)
      let actions = ActionSet.elements under_explored in
      let idx = Random.int (List.length actions) in
      List.nth actions idx
    end

  let calculate_reward values action =
    (* Use last node as reward *)
    let n = Tensor.size values 0 - 1 in
    Tensor.float_value values.(.n)

  let update_regret state observation =
    (* Calculate regret against optimal action *)
    let optimal_reward = 
      Linear_sem.simulate state.model (select_best_action state) 
                         observation.values
      |> fun x -> x.(.(state.model.dim - 1))
      |> Tensor.float_value
    in
    state.cumulative_regret +. (optimal_reward -. observation.reward)

  let update state observation =
    let regret = update_regret state observation in
    
    (* Check if we should refine action set *)
    let state = 
      if state.t mod state.config.min_exploration_rounds = 0 then
        refine_actions state 
      else 
        state
    in
    
    { state with
      t = state.t + 1;
      observations = observation :: state.observations;
      cumulative_regret = regret
    }
end

let create graph model eta horizon ?(config=default_config) () =
  let t1, t2 = StructureLearning.calculate_exploration_rounds 
    model.dim eta config.epsilon 
  in
  let exploration_rounds = max t1 config.min_exploration_rounds in
  
  let max_simultaneous = int_of_float (sqrt (float model.dim)) in
  let all_actions = ActionSet.singleton NodeSet.empty in
  
  (* Generate all possible intervention combinations up to max_simultaneous *)
  let all_actions = 
    let rec combinations n k acc current start =
      if k = 0 then ActionSet.add current acc
      else if start >= n then acc
      else
        let with_curr = combinations n (k-1) acc 
          (NodeSet.add start current) (start+1) in
        let without = combinations n k acc current (start+1) in
        with_curr |> fun acc -> without 
    in
    let rec build_actions k acc =
      if k > max_simultaneous then acc
      else build_actions (k+1) (combinations model.dim k acc NodeSet.empty 0)
    in
    build_actions 1 all_actions
  in
  
  {
    graph;
    model;
    t = 0;
    observations = [];
    exploration_rounds;
    eta;
    config = { config with phases = int_of_float (log2 (sqrt (float horizon))) };
    current_phase = 1;
    candidate_actions = all_actions;
    cumulative_regret = 0.0;
  }

let run state horizon =
  let rec loop state =
    if state.t >= horizon then
      state
    else begin
      (* Select and apply intervention *)
      let action = InterventionDesign.select_action state in
      let values = Linear_sem.simulate state.model action (Tensor.randn [state.model.dim]) in
      
      (* Record observation with reward *)
      let observation = {
        values;
        intervention = action;
        timestamp = state.t;
        reward = InterventionDesign.calculate_reward values action;
      } in
      
      (* Update state *)
      let state = InterventionDesign.update state observation in
      loop state
    end
  in
  
  let initial_state = 
    if state.t = 0 then
      (* Run structure learning first *)
      let parents = StructureLearning.learn state in
      let graph = Graph.create state.graph.nodes parents in
      { state with graph }
    else
      state
  in
  
  loop initial_state