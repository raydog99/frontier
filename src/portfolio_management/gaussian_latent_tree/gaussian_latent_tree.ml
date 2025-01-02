open Torch

module Tree = struct
  type node = 
    | Leaf of int
    | Internal of int

  type edge = {
    source: node;
    target: node;
    correlation: float Tensor.t;
  }

  type t = {
    nodes: node list;
    edges: edge list;
    n_leaves: int;
    n_internal: int;
  }

  let create nodes edges = 
    let n_leaves = List.length (List.filter (function Leaf _ -> true | _ -> false) nodes) in
    let n_internal = List.length (List.filter (function Internal _ -> true | _ -> false) nodes) in
    { nodes; edges; n_leaves; n_internal }

  let get_neighbors node tree =
    List.filter_map (fun edge ->
      if edge.source = node then Some edge.target
      else if edge.target = node then Some edge.source
      else None
    ) tree.edges

  let get_edge source target tree =
    List.find_opt (fun edge ->
      (edge.source = source && edge.target = target) ||
      (edge.source = target && edge.target = source)
    ) tree.edges

  let check_degree_constraints tree =
    List.for_all (function
      | Tree.Leaf _ -> true
      | Tree.Internal _ as node ->
          List.length (Tree.get_neighbors node tree) >= 3
    ) tree.nodes

  let normalize_tree tree =
    (* Create new tree with all internal nodes having degree >= 3 *)
    let rec merge_degree_two_nodes current_tree =
      let needs_merge = ref false in
      List.iter (fun node ->
        match node with
        | Tree.Internal _ ->
            if List.length (Tree.get_neighbors node current_tree) = 2 then
              needs_merge := true
        | _ -> ()
      ) current_tree.nodes;
      
      if !needs_merge then
        (* Find a degree-2 internal node and merge it *)
        let merged_tree = ref current_tree in
        List.iter (fun node ->
          match node with
          | Tree.Internal i as n ->
              if List.length (Tree.get_neighbors n !merged_tree) = 2 then begin
                let neighbors = Tree.get_neighbors n !merged_tree in
                match neighbors with
                | [n1; n2] ->
                    (* Merge node with its neighbors *)
                    let new_edges = 
                      List.filter (fun edge ->
                        edge.Tree.source <> n && edge.Tree.target <> n
                      ) !merged_tree.edges in
                    let new_edge = {
                      Tree.source = n1;
                      target = n2;
                      correlation = Tensor.ones [1]  (* Update correlation appropriately *)
                    } in
                    let new_nodes = 
                      List.filter (fun node -> node <> n) !merged_tree.nodes in
                    merged_tree := 
                      Tree.create new_nodes (new_edge :: new_edges)
              end
          | _ -> ()
        ) current_tree.nodes;
        merge_degree_two_nodes !merged_tree
      else
        current_tree
    in
    merge_degree_two_nodes tree

  let partition_leaves node tree =
    let rec collect_leaves visited current =
      let leaves = ref [] in
      let neighbors = Tree.get_neighbors current tree in
      List.iter (fun neighbor ->
        if not (List.mem neighbor visited) then
          match neighbor with
          | Tree.Leaf i -> leaves := i :: !leaves
          | _ -> 
              leaves := 
                (collect_leaves (current :: visited) neighbor) @ !leaves
      ) neighbors;
      !leaves
    in
    collect_leaves [] node

  let compute_path_product params leaf_idx tree =
    let rec compute_product current target acc visited =
      if current = target then acc
      else
        let neighbors = Tree.get_neighbors current tree in
        let unvisited = 
          List.filter (fun n -> not (List.mem n visited)) neighbors in
        match unvisited with
        | [] -> acc
        | next :: _ ->
            let edge_corr = 
              match Tree.get_edge current next tree with
              | Some edge -> Tensor.get edge.correlation [0]
              | None -> 1.0
            in
            compute_product next target (acc *. edge_corr) (current :: visited)
    in
    match Tree.get_neighbors (Tree.Leaf leaf_idx) tree with
    | [] -> 1.0
    | first :: _ -> compute_product (Tree.Leaf leaf_idx) first 1.0 []

  let get_subtree root tree =
    let rec collect_nodes visited current =
      let subtree_nodes = ref [current] in
      let subtree_edges = ref [] in
      let neighbors = Tree.get_neighbors current tree in
      
      List.iter (fun neighbor ->
        if not (List.mem neighbor visited) then begin
          subtree_nodes := neighbor :: !subtree_nodes;
          match Tree.get_edge current neighbor tree with
          | Some edge -> subtree_edges := edge :: !subtree_edges
          | None -> ()
        end
      ) neighbors;
      
      List.iter (fun neighbor ->
        if not (List.mem neighbor visited) then begin
          let (nodes, edges) = collect_nodes (current :: visited) neighbor in
          subtree_nodes := nodes @ !subtree_nodes;
          subtree_edges := edges @ !subtree_edges
        end
      ) neighbors;
      
      (!subtree_nodes, !subtree_edges)
    in
    
    let (nodes, edges) = collect_nodes [] root in
    Tree.create nodes edges

  let is_connected tree =
    match tree.nodes with
    | [] -> true
    | first :: _ ->
        let rec visit visited node =
          let neighbors = Tree.get_neighbors node tree in
          let unvisited = List.filter (fun n -> not (List.mem n visited)) neighbors in
          List.fold_left 
            (fun acc neighbor -> acc @ visit (node :: visited) neighbor)
            [node]
            unvisited
        in
        let visited = visit [] first in
        List.length visited = List.length tree.nodes

  let get_path_length source target tree =
    match find_path source target tree with
    | None -> None
    | Some path -> Some (List.length path - 1)
end

let inverse mat =
  let n = Tensor.size mat 0 in
  let augmented = Tensor.cat [mat; Tensor.eye n] 1 in
  let result = Tensor.zeros [n; n] in
  
  let eliminate mat =
    let m = ref (Tensor.copy mat) in
    for i = 0 to n - 1 do
      let pivot = Tensor.get !m [i; i] in
      for j = 0 to (2 * n - 1) do
        Tensor.set !m [i; j] (Tensor.get !m [i; j] /. pivot)
      done;
      for k = 0 to n - 1 do
        if k <> i then
          let factor = Tensor.get !m [k; i] in
          for j = 0 to (2 * n - 1) do
            let curr = Tensor.get !m [k; j] in
            let subtract = factor *. (Tensor.get !m [i; j]) in
            Tensor.set !m [k; j] (curr -. subtract)
          done
      done
    done;
    !m
  in
  
  let eliminated = eliminate augmented in
  for i = 0 to n - 1 do
    for j = 0 to n - 1 do
      Tensor.set result [i; j] (Tensor.get eliminated [i; (j + n)])
    done
  done;
  result

let compute_jacobian p u =
  let n = Tensor.size u 0 in
  let j = Tensor.zeros [n; n] in
  for i = 0 to n - 1 do
    for k = 0 to n - 1 do
      if i = k then
        let sum = Tensor.sum (Tensor.slice u [Some k+1; None]) [] in
        Tensor.set j [i; k] (Tensor.get sum [])
      else
        Tensor.set j [i; k] (Tensor.get u [i])
    done
  done;
  j

let condition_gaussian mean cov obs_idx =
  let n = Tensor.size mean 0 in
  let n_obs = List.length obs_idx in
  let n_hidden = n - n_obs in
  let hidden_idx = List.init n (fun i -> i) |> 
                  List.filter (fun i -> not (List.mem i obs_idx)) in
  
  (* Split covariance matrix *)
  let sigma_11 = Tensor.zeros [n_obs; n_obs] in  (* observed-observed *)
  let sigma_12 = Tensor.zeros [n_obs; n_hidden] in  (* observed-hidden *)
  let sigma_21 = Tensor.zeros [n_hidden; n_obs] in  (* hidden-observed *)
  let sigma_22 = Tensor.zeros [n_hidden; n_hidden] in  (* hidden-hidden *)
  
  (* Fill submatrices *)
  List.iteri (fun i idx1 ->
    List.iteri (fun j idx2 ->
      Tensor.set sigma_11 [i; j] (Tensor.get cov [idx1; idx2])
    ) obs_idx;
    List.iteri (fun j idx2 ->
      Tensor.set sigma_12 [i; j] (Tensor.get cov [idx1; idx2])
    ) hidden_idx
  ) obs_idx;
  
  List.iteri (fun i idx1 ->
    List.iteri (fun j idx2 ->
      Tensor.set sigma_21 [i; j] (Tensor.get cov [idx1; idx2])
    ) obs_idx;
    List.iteri (fun j idx2 ->
      Tensor.set sigma_22 [i; j] (Tensor.get cov [idx1; idx2])
    ) hidden_idx
  ) hidden_idx;
  
  (* Compute conditional covariance using Schur complement *)
  let sigma_11_inv = MatrixOps.inverse sigma_11 in
  let schur = Tensor.sub sigma_22 
    (Tensor.matmul (Tensor.matmul sigma_21 sigma_11_inv) sigma_12) in
  schur

let compute_conditional_mean params observations i j =
  let n = Tensor.size observations 1 in
  let joint_cov = Tensor.zeros [n; n] in
  
  (* Build joint covariance matrix *)
  for k = 0 to n - 1 do
    for l = 0 to n - 1 do
      let cov = 
        if k = l then Tensor.get params.variances [k]
        else 
          let rho = Tensor.get params.correlations [k; l] in
          rho *. sqrt(Tensor.get params.variances [k] *. 
                     Tensor.get params.variances [l])
      in
      Tensor.set joint_cov [k; l] cov
    done
  done;
  
  (* Condition on observation i to get mean for j *)
  let obs_idx = [i] in
  let cond_cov = condition_gaussian (Tensor.zeros [n]) joint_cov obs_idx in
  let sigma_12 = Tensor.slice joint_cov 
    [Some j; Some i; Some (j+1); Some (i+1)] in
  let sigma_11 = Tensor.slice joint_cov 
    [Some i; Some i; Some (i+1); Some (i+1)] in
  
  Tensor.matmul sigma_12 (MatrixOps.inverse sigma_11)

let compute_conditional_variance params i tree =
  let n = List.length tree.nodes in
  let joint_cov = Tensor.zeros [n; n] in
  
  (* Build joint covariance *)
  List.iter (fun edge ->
    let k = match edge.Tree.source with Tree.Leaf n | Tree.Internal n -> n in
    let l = match edge.Tree.target with Tree.Leaf n | Tree.Internal n -> n in
    let rho = Tensor.get edge.correlation [0] in
    let cov = rho *. sqrt(Tensor.get params.variances [k] *. 
                         Tensor.get params.variances [l]) in
    Tensor.set joint_cov [k; l] cov;
    Tensor.set joint_cov [l; k] cov
  ) tree.edges;
  
  (* Add diagonal terms *)
  for k = 0 to n - 1 do
    Tensor.set joint_cov [k; k] (Tensor.get params.variances [k])
  done;
  
  (* Get observed indices (all except i) *)
  let obs_idx = List.init n (fun k -> k) |> List.filter (fun k -> k <> i) in
  condition_gaussian (Tensor.zeros [n]) joint_cov obs_idx

let compute_conditional_expectation params i j tree =
  let cond_mean = compute_conditional_mean params params.variances i j in
  Tensor.mean cond_mean [] |> Tensor.get []

let compute_conditional_expectation_squared params i tree =
  let cond_var = compute_conditional_variance params i tree in
  let cond_mean = compute_conditional_mean params params.variances i i in
  let mean_sq = Tensor.mul cond_mean cond_mean in
  Tensor.add mean_sq cond_var
  |> Tensor.mean []
  |> Tensor.get []

let compute_information_parameter params node_idx tree =
  (* Compute precision matrix *)
  let precision = MatrixOps.inverse 
    (compute_conditional_variance params node_idx tree) in
  
  (* Extract information parameter (h) for the node *)
  let n = List.length tree.nodes in
  let h = Tensor.zeros [n] in
  
  List.iteri (fun i node ->
    match node with
    | Tree.Leaf idx ->
        let contrib = Tensor.get precision [node_idx; idx] *. 
                     Tensor.get params.variances [idx] in
        let curr = Tensor.get h [node_idx] in
        Tensor.set h [node_idx] (curr +. contrib)
    | _ -> ()
  ) tree.nodes;
  
  Tensor.get h [node_idx]

let compute_leaf_linear_combination params partition tree =
  (* Compute weights for each leaf in partition *)
  let weights = Array.make (List.length partition) 0. in
  
  List.iteri (fun i leaf_idx ->
    (* Compute path product to root *)
    let path_prod = ref 1. in
    let rec compute_to_root current =
      match Tree.get_neighbors current tree with
      | [] -> ()
      | parent :: _ ->
          (match Tree.get_edge current parent tree with
           | Some edge -> 
               path_prod := !path_prod *. Tensor.get edge.correlation [0]
           | None -> ());
          compute_to_root parent
    in
    compute_to_root (Tree.Leaf leaf_idx);
    weights.(i) <- !path_prod
  ) partition;
  
  (* Compute linear combination *)
  let result = ref 0. in
  List.iteri (fun i leaf_idx ->
    result := !result +. weights.(i) *. 
      Tensor.get params.variances [leaf_idx]
  ) partition;
  !result

let compute_conditional_covariance params node_indices tree =
  let n = List.length tree.nodes in
  let cov = Tensor.zeros [n; n] in
  
  List.iter (fun i ->
    List.iter (fun j ->
      let rho = 
        if i = j then 1.0
        else
          match Tree.get_edge (Tree.Internal i) (Tree.Internal j) tree with
          | Some edge -> Tensor.get edge.correlation [0]
          | None -> 0.0
      in
      Tensor.set cov [i; j] rho
    ) node_indices
  ) node_indices;
  cov

let compute_precision_matrix params tree =
  let n = List.length tree.nodes in
  let precision = Tensor.zeros [n; n] in
  
  (* Set precision matrix entries based on tree structure *)
  List.iter (fun edge ->
    let i = match edge.Tree.source with Tree.Leaf n | Tree.Internal n -> n in
    let j = match edge.Tree.target with Tree.Leaf n | Tree.Internal n -> n in
    let rho = Tensor.get edge.correlation [0] in
    let sigma_i = Tensor.get params.variances [i] in
    let sigma_j = Tensor.get params.variances [j] in
    
    let prec_val = -. rho /. (sqrt (sigma_i *. sigma_j)) in
    Tensor.set precision [i; j] prec_val;
    Tensor.set precision [j; i] prec_val
  ) tree.edges;
  
  (* Set diagonal entries *)
  for i = 0 to n - 1 do
    let sigma_i = Tensor.get params.variances [i] in
    let sum_neighbors = ref 0. in
    List.iter (fun edge ->
      let is_neighbor = 
        match (edge.source, edge.target) with
        | (Tree.Leaf m, _) | (_, Tree.Leaf m) | 
          (Tree.Internal m, _) | (_, Tree.Internal m) when m = i -> true
        | _ -> false
      in
      if is_neighbor then
        sum_neighbors := !sum_neighbors +. 
          Float.abs (Tensor.get precision [i; 
            (match edge.source with 
             | Tree.Leaf m | Tree.Internal m when m = i -> 
                 match edge.target with 
                 | Tree.Leaf n | Tree.Internal n -> n
             | _ -> 
                 match edge.source with 
                 | Tree.Leaf n | Tree.Internal n -> n)])
    ) tree.edges;
    Tensor.set precision [i; i] (1. /. sigma_i +. !sum_neighbors)
  done;
  precision

let compute_marginal_distribution params indices tree =
  let n = List.length indices in
  let mean = Tensor.zeros [n] in
  let cov = Tensor.zeros [n; n] in
  
  (* Compute covariance matrix for marginal distribution *)
  List.iteri (fun i idx1 ->
    List.iteri (fun j idx2 ->
      let sigma_1 = Tensor.get params.variances [idx1] in
      let sigma_2 = Tensor.get params.variances [idx2] in
      
      let rho = 
        if idx1 = idx2 then 1.0
        else
          match Tree.get_edge 
            (Tree.Leaf idx1) (Tree.Leaf idx2) tree with
          | Some edge -> Tensor.get edge.correlation [0]
          | None ->
              (* Compute correlation along path *)
              match TreeOps.find_path 
                (Tree.Leaf idx1) (Tree.Leaf idx2) tree with
              | Some path ->
                  List.fold_left2 
                    (fun acc node1 node2 ->
                      match Tree.get_edge node1 node2 tree with
                      | Some edge -> 
                          acc *. Tensor.get edge.correlation [0]
                      | None -> acc
                    ) 1. path (List.tl path)
              | None -> 0.0
      in
      
      let cov_val = rho *. sqrt (sigma_1 *. sigma_2) in
      Tensor.set cov [i; j] cov_val
    ) indices
  ) indices;
  
  (mean, cov)

module EM = struct
  type params = {
    correlations: float Tensor.t;
    variances: float Tensor.t;
  }

  let create_params n =
    let correlations = Tensor.ones [n; n] in
    let variances = Tensor.ones [n] in
    { correlations; variances }

  let e_step params observations tree =
    let n_samples = Tensor.size observations 0 in
    let n_total = tree.n_leaves + tree.n_internal in
    
    let cond_mean = Tensor.zeros [n_samples; tree.n_internal] in
    let cond_cov = Tensor.zeros [tree.n_internal; tree.n_internal] in
    
    (* Build precision matrix *)
    let precision = Tensor.zeros [n_total; n_total] in
    List.iter (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      let rho = Tensor.get edge.correlation [0] in
      Tensor.set precision [i; j] (-.rho);
      Tensor.set precision [j; i] (-.rho)
    ) tree.edges;
    
    (* Add diagonal terms *)
    for i = 0 to n_total - 1 do
      Tensor.set precision [i; i] 1.0
    done;

    (cond_mean, cond_cov)

  let m_step observations posteriors tree =
    let params = create_params (List.length tree.nodes) in
    
    (* Update correlations *)
    List.iter (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      
      let new_corr = Tensor.zeros [1] in
      Tensor.set edge.correlation [0] (Tensor.get new_corr [0])
    ) tree.edges;
    
    params

  let fit ?(max_iter=100) ?(tol=1e-6) initial_params observations tree =
    let rec iterate params iter =
      if iter >= max_iter then params
      else
        let posteriors = e_step params observations tree in
        let new_params = m_step observations posteriors tree in
        
        let diff = Tensor.sub new_params.correlations params.correlations 
                  |> Tensor.abs 
                  |> Tensor.mean in
        if Tensor.get diff [] < tol then new_params
        else iterate new_params (iter + 1)
    in
    iterate initial_params 0

  let update_parameters params tree =
    let n_nodes = List.length tree.nodes in
    let new_correlations = Tensor.zeros [n_nodes; n_nodes] in
    
    List.iter (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      
      let lambda = Tensor.zeros [1] in
      let delta = Tensor.zeros [1] in
      let new_rho = Tensor.zeros [1] in
      
      Tensor.set new_correlations [i; j] (Tensor.get new_rho [0]);
      Tensor.set new_correlations [j; i] (Tensor.get new_rho [0])
    ) tree.edges;
    
    { correlations = new_correlations; variances = params.variances }

  let fit ?(max_iter=100) ?(tol=1e-6) initial_params tree =
    let rec iterate params iter =
      if iter >= max_iter then params
      else
        let new_params = update_parameters params tree in
        let diff = Tensor.sub new_params.correlations params.correlations 
                  |> Tensor.abs 
                  |> Tensor.mean in
        if Tensor.get diff [] < tol then new_params
        else iterate new_params (iter + 1)
    in
    iterate initial_params 0

  let update_parameters params tree observations =
    let n_nodes = List.length tree.nodes in
    let new_params = create_params n_nodes in
    
    List.iter (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      
      (* Compute lambda coefficients *)
      let lambdas = Tensor.zeros [n_nodes] in
      List.iteri (fun k node ->
        let rho = match Tree.get_edge edge.source node tree with
          | Some e -> Tensor.get e.correlation [0]
          | None -> 0. in
        let denom = 1. -. (rho *. rho) in
        Tensor.set lambdas [k] (rho /. denom)
      ) tree.nodes;
      
      (* Compute delta terms *)
      let deltas = Tensor.zeros [n_nodes; n_nodes] in
      List.iteri (fun k1 n1 ->
        List.iteri (fun k2 n2 ->
          if k1 <> k2 then
            let rho_star = match Tree.get_edge n1 n2 tree with
              | Some e -> Tensor.get e.correlation [0]
              | None -> 0. in
            let rho = Tensor.get params.correlations [k1; k2] in
            Tensor.set deltas [k1; k2] (rho_star -. rho)
        ) tree.nodes
      ) tree.nodes;
      
      (* Update correlation *)
      let numerator = Tensor.get params.correlations [i; j] +.
                     (Tensor.sum (Tensor.mul deltas lambdas) [] |> Tensor.get []) in
      let denominator = 1. +.
                       (Tensor.sum (Tensor.mul deltas (Tensor.mul lambdas lambdas)) []
                        |> Tensor.get []) in
      let new_rho = numerator /. denominator in
      
      Tensor.set new_params.correlations [i; j] new_rho;
      Tensor.set new_params.correlations [j; i] new_rho
    ) tree.edges;
    
    new_params

  let is_trivial_point params tree =
    List.exists (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      let rho = Tensor.get params.correlations [i; j] in
      Float.abs rho < 1e-10 || Float.abs (1. -. rho) < 1e-10
    ) tree.edges

  let fit_with_guarantees ?(max_iter=100) ?(tol=1e-6) ?(alpha=0.1) ?(beta=0.1) 
      initial_params observations tree =
    let rec iterate params iter prev_ll =
      if iter >= max_iter then params
      else
        let new_params = update_parameters params tree observations in
        let new_ll = GeneralTreeModel.compute_likelihood 
          {tree; 
           node_params = Array.make (List.length tree.nodes) 
             {mean = Tensor.zeros [1]; variance = Tensor.ones [1]};
           edge_correlations = new_params.correlations} 
          observations in
        
        if Float.abs (new_ll -. prev_ll) < tol then new_params
        else if is_trivial_point new_params tree then params
        else iterate new_params (iter + 1) new_ll
    in
    
    let initial_ll = GeneralTreeModel.compute_likelihood 
      {tree; 
       node_params = Array.make (List.length tree.nodes) 
         {mean = Tensor.zeros [1]; variance = Tensor.ones [1]};
       edge_correlations = initial_params.correlations} 
      observations in
    iterate initial_params 0 initial_ll
end

module ConvergenceAnalysis = struct
  type convergence_state = {
    correlations: float Tensor.t;
    iteration: int;
    log_likelihood: float;
    stationary: bool;
  }

  let check_stationarity params next_params tol =
    let diff = Tensor.sub next_params.correlations params.correlations
               |> Tensor.abs
               |> Tensor.mean
               |> Tensor.get [] in
    diff < tol

  let characterize_stationary_points params tree =
    let n = List.length tree.nodes in
    let special_points = Array.make n false in
    
    (* Check for trivial fixed points *)
    let is_trivial = ref false in
    List.iter (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      let rho = Tensor.get params.correlations [i; j] in
      if Float.abs rho < 1e-10 || Float.abs (1. -. rho) < 1e-10 then begin
        is_trivial := true;
        special_points.(i) <- true;
        special_points.(j) <- true
      end
    ) tree.edges;
    (!is_trivial, special_points)

  let verify_convergence_bounds params =
    let n = Tensor.size params.correlations 0 in
    let valid = ref true in
    
    for i = 0 to n - 1 do
      for j = 0 to n - 1 do
        let rho = Tensor.get params.correlations [i; j] in
        if rho <= 0. || rho >= 1. then
          valid := false
      done
    done;
    !valid

  let verify_no_interior_stationary params tree true_params =
    let n = List.length tree.nodes in
    let is_interior = ref true in
    let is_different = ref false in
    
    (* Check if point is in interior *)
    List.iter (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      let rho = Tensor.get params.correlations [i; j] in
      if rho <= 0. || rho >= 1. then
        is_interior := false
    ) tree.edges;

    (* Check if different from true parameters *)
    if !is_interior then
      List.iter (fun edge ->
        let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
        let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
        let rho = Tensor.get params.correlations [i; j] in
        let true_rho = Tensor.get true_params.correlations [i; j] in
        if Float.abs (rho -. true_rho) > 1e-10 then
          is_different := true
      ) tree.edges;

    !is_interior && !is_different

  let analyze_convergence ?(max_iter=1000) ?(tol=1e-6) 
      params true_params observations tree =
    let rec iterate state =
      if state.iteration >= max_iter then 
        Error "Max iterations exceeded"
      else
        let next_params = PopulationEM.update_parameters 
          {correlations = state.correlations; variances = params.variances} tree in
          
        if check_stationarity params next_params tol then
          Ok next_params
        else
          let new_ll = GeneralTreeModel.compute_likelihood 
            {tree; 
             node_params = Array.make (List.length tree.nodes) 
               {mean = Tensor.zeros [1]; variance = Tensor.ones [1]};
             edge_correlations = next_params.correlations} 
            observations in
            
          iterate {
            correlations = next_params.correlations;
            iteration = state.iteration + 1;
            log_likelihood = new_ll;
            stationary = true
          }
    in
    
    iterate {
      correlations = params.correlations;
      iteration = 0;
      log_likelihood = Float.neg_infinity;
      stationary = false
    }
end

module TaylorSeriesAnalysis = struct
  type expansion_terms = {
    constant: float;
    first_order: float Tensor.t;
    second_order: float Tensor.t;
  }

  let compute_derivatives params tree idx =
    let n = List.length tree.nodes in
    let first_deriv = Tensor.zeros [n] in
    let second_deriv = Tensor.zeros [n; n] in
    
    List.iter (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      
      if i = idx || j = idx then begin
        let rho = Tensor.get params.correlations [i; j] in
        let other_idx = if i = idx then j else i in
        let deriv = 
          if Float.abs rho < 1e-10 then
            let rho_star = Tensor.get params.correlations [i; j] in
            rho_star *. (1. -. rho_star *. rho_star)
          else
            1. /. (1. -. rho *. rho)
        in
        Tensor.set first_deriv [other_idx] deriv
      end
    ) tree.edges;

    (* Second derivatives *)
    List.iter (fun edge1 ->
      let i1 = match edge1.source with Tree.Leaf n | Tree.Internal n -> n in
      let j1 = match edge1.target with Tree.Leaf n | Tree.Internal n -> n in
      
      List.iter (fun edge2 ->
        let i2 = match edge2.source with Tree.Leaf n | Tree.Internal n -> n in
        let j2 = match edge2.target with Tree.Leaf n | Tree.Internal n -> n in
        
        if (i1 = idx || j1 = idx) && (i2 = idx || j2 = idx) then begin
          let rho1 = Tensor.get params.correlations [i1; j1] in
          let rho2 = Tensor.get params.correlations [i2; j2] in
          
          let other_idx1 = if i1 = idx then j1 else i1 in
          let other_idx2 = if i2 = idx then j2 else i2 in
          
          let deriv = 
            if Float.abs rho1 < 1e-10 && Float.abs rho2 < 1e-10 then
              -2. *. rho1 *. rho2
            else
              2. *. rho1 *. rho2 /. ((1. -. rho1 *. rho1) *. (1. -. rho2 *. rho2))
          in
          Tensor.set second_deriv [other_idx1; other_idx2] deriv
        end
      ) tree.edges
    ) tree.edges;
    
    {constant = Tensor.get params.correlations [idx; idx];
     first_order = first_deriv;
     second_order = second_deriv}

  let analyze_near_zero params tree =
    let n = List.length tree.nodes in
    let repulsion_strength = Array.make n 0. in
    
    List.iteri (fun idx _ ->
      let terms = compute_derivatives params tree idx in
      let linear_term = Tensor.sum terms.first_order [] |> Tensor.get [] in
      let quadratic_term = 
        Tensor.sum (Tensor.sum terms.second_order [0]) [] 
        |> Tensor.get [] in
      repulsion_strength.(idx) <- linear_term -. 0.5 *. Float.abs quadratic_term
    ) tree.nodes;
    
    Array.for_all (fun strength -> strength > 0.) repulsion_strength

  let analyze_near_special_point params tree i =
    let n = List.length tree.nodes in
    let stability = Array.make n false in
    
    let special_point = EM.create_params n in
    List.iter (fun edge ->
      let src = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let tgt = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      
      let rho = 
        if src = i || tgt = i then 1.0
        else begin
          let rho_src = Tensor.get params.correlations [src; i] in
          let rho_tgt = Tensor.get params.correlations [tgt; i] in
          rho_src *. rho_tgt
        end
      in
      Tensor.set special_point.correlations [src; tgt] rho;
      Tensor.set special_point.correlations [tgt; src] rho
    ) tree.edges;
    
    List.iteri (fun idx _ ->
      let terms = compute_derivatives special_point tree idx in
      let max_eigenval = 
        Tensor.eigenvals terms.second_order 
        |> Tensor.max [] 
        |> Tensor.get [] in
      stability.(idx) <- max_eigenval < 0.
    ) tree.nodes;
    
    Array.exists not stability

  let analyze_convergence params tree =
    let repels_zero = analyze_near_zero params tree in
    let special_points_unstable = 
      List.init (List.length tree.nodes) (fun i ->
        analyze_near_special_point params tree i
      ) in
    
    if repels_zero && List.for_all (fun unstable -> unstable) special_points_unstable 
    then Ok params
    else Error "Convergence conditions not satisfied"

  let verify_taylor_accuracy params tree =
    let n = List.length tree.nodes in
    let accuracy = ref true in
    
    List.iteri (fun idx _ ->
      let terms = compute_derivatives params tree idx in
      let next_params = PopulationEM.update_parameters params tree in
      
      let predicted = 
        terms.constant +.
        (Tensor.sum terms.first_order [] |> Tensor.get []) +.
        0.5 *. (Tensor.sum (Tensor.sum terms.second_order [0]) [] |> Tensor.get []) in
      
      let actual = Tensor.get next_params.correlations [idx; idx] in
      
      if Float.abs (predicted -. actual) > 1e-6 then
        accuracy := false
    ) tree.nodes;
    
    !accuracy
end

module GeneralTreeModel = struct
  type node_params = {
    mean: float Tensor.t;
    variance: float Tensor.t;
  }

  type model = {
    tree: Tree.t;
    node_params: node_params array;
    edge_correlations: float Tensor.t;
  }

  let create tree =
    let n = List.length tree.nodes in
    let node_params = Array.init n (fun _ -> 
      {mean = Tensor.zeros [1]; variance = Tensor.ones [1]}) in
    let edge_correlations = Tensor.ones [n; n] in
    {tree; node_params; edge_correlations}

  let compute_likelihood model observations =
    let n_samples = Tensor.size observations 0 in
    let log_likelihood = ref 0. in
    
    for i = 0 to n_samples - 1 do
      let obs_values = Tensor.slice observations [Some i; None] in
      let n = Array.length model.node_params in
      let joint_cov = Tensor.zeros [n; n] in
      
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          if i = j then
            Tensor.set joint_cov [i; j] (Tensor.get model.node_params.(i).variance [0])
          else
            match TreeOps.find_path (Tree.Leaf i) (Tree.Leaf j) model.tree with
            | Some path ->
                let prob = TreeOps.compute_path_product 
                  {correlations = model.edge_correlations;
                   variances = Tensor.ones [n]} i model.tree in
                let cov = prob *. 
                  (sqrt (Tensor.get model.node_params.(i).variance [0]) *.
                   sqrt (Tensor.get model.node_params.(j).variance [0])) in
                Tensor.set joint_cov [i; j] cov
            | None -> ()
        done
      done;
      
      let obs_idx = List.init model.tree.n_leaves (fun i -> i) in
      let cond_cov = condition_gaussian 
        (Tensor.zeros [n]) joint_cov obs_idx in
      let obs_cov = Tensor.slice joint_cov 
        [Some 0; Some model.tree.n_leaves; Some 0; Some model.tree.n_leaves] in
      
      log_likelihood := !log_likelihood +. (
        -0.5 *. (Float.log (2. *. Float.pi) *. float model.tree.n_leaves +.
                (Tensor.logdet obs_cov |> Tensor.get []) +.
                (Tensor.matmul 
                  (Tensor.matmul obs_values (MatrixOps.inverse obs_cov)) 
                  (Tensor.transpose obs_values 0 1)
                |> Tensor.get [0; 0]))
      )
    done;
    !log_likelihood

  let em_step model observations =
    let n = Array.length model.node_params in
    let new_model = create model.tree in
    let sufficient_stats = Tensor.zeros [n; n] in
    
    for i = 0 to (Tensor.size observations 0) - 1 do
      let obs = Tensor.slice observations [Some i; None] in
      let obs_idx = List.init model.tree.n_leaves (fun i -> i) in
      
      let joint_mean = Tensor.zeros [n] in
      let joint_cov = Tensor.zeros [n; n] in
      
      List.iter (fun node1 ->
        List.iter (fun node2 ->
          match (node1, node2) with
          | Tree.Internal i1, Tree.Internal i2 ->
              (match TreeOps.find_path node1 node2 model.tree with
               | Some path ->
                   let prob = TreeOps.compute_path_product 
                     {correlations = model.edge_correlations;
                      variances = Tensor.ones [n]} i1 model.tree in
                   Tensor.set joint_cov [i1; i2] prob
               | None -> ())
          | _ -> ()
        ) model.tree.nodes
      ) model.tree.nodes;
      
      let cond_cov = condition_gaussian joint_mean joint_cov obs_idx in
      Tensor.add_ sufficient_stats 
        (Tensor.add joint_cov 
          (Tensor.matmul 
            (Tensor.transpose joint_mean 0 1) joint_mean));
    done;
    
    List.iter (fun edge ->
      let i = match edge.source with Tree.Leaf n | Tree.Internal n -> n in
      let j = match edge.target with Tree.Leaf n | Tree.Internal n -> n in
      
      let new_corr = 
        Tensor.get sufficient_stats [i; j] /.
        (sqrt (Tensor.get sufficient_stats [i; i]) *.
         sqrt (Tensor.get sufficient_stats [j; j])) in
      
      Tensor.set new_model.edge_correlations [i; j] new_corr;
      Tensor.set new_model.edge_correlations [j; i] new_corr;
    ) model.tree.edges;
    
    new_model

  let fit ?(max_iter=100) ?(tol=1e-6) initial_model observations =
    let rec iterate model iter best_ll =
      if iter >= max_iter then model
      else
        let new_model = em_step model observations in
        let new_ll = compute_likelihood new_model observations in
        
        if Float.abs (new_ll -. best_ll) < tol then new_model
        else iterate new_model (iter + 1) new_ll
    in
    
    let initial_ll = compute_likelihood initial_model observations in
    iterate initial_model 0 initial_ll
end

module GeneralTreeAnalysis = struct
  type sufficient_statistics = {
    first_moments: float Tensor.t;
    second_moments: float Tensor.t;
    empirical_cov: float Tensor.t;
  }

  let verify_conditional_conservation params tree =
    let verify_covariance old_dist new_dist = 
      let n = Tensor.size old_dist 0 in
      let max_diff = ref 0. in
      
      for i = 0 to n - 1 do
        for j = 0 to n - 1 do
          let old_cov = Tensor.get old_dist [i; j] in
          let new_cov = Tensor.get new_dist [i; j] in
          max_diff := max !max_diff (Float.abs (old_cov -. new_cov))
        done
      done;
      !max_diff < 1e-6
    in
    
    let next_params = PopulationEM.update_parameters params tree in
    verify_covariance params.correlations next_params.correlations

  let verify_likelihood_stationarity params tree observations =
    let next_params = PopulationEM.update_parameters params tree in
    let diff = Tensor.sub next_params.correlations params.correlations
               |> Tensor.abs
               |> Tensor.mean
               |> Tensor.get [] in
    diff < 1e-6

  let verify_conditional_expectations params params_t tree =
    let n = List.length tree.nodes in
    let valid = ref true in
    
    List.iter (fun node1 ->
      match node1 with
      | Tree.Internal i ->
          List.iter (fun node2 ->
            match node2 with
            | Tree.Internal j ->
                let exp1 = compute_conditional_expectation 
                  params i j tree in
                let exp2 = compute_conditional_expectation 
                  params_t i j tree in
                if Float.abs (exp1 -. exp2) > 1e-6 then
                  valid := false
            | _ -> ()
          ) tree.nodes
      | _ -> ()
    ) tree.nodes;
    !valid

  let verify_linear_combination_property params tree =
    let valid = ref true in
    
    List.iter (fun node ->
      match node with
      | Tree.Internal y ->
          let leaf_partition = TreeOps.partition_leaves node tree in
          let h = compute_information_parameter params y tree in
          let linear_comb = compute_leaf_linear_combination 
            params leaf_partition tree in
          if Float.abs (h -. linear_comb) > 1e-6 then
            valid := false
      | _ -> ()
    ) tree.nodes;
    !valid

  let verify_invertibility_conditions params tree =
    let valid = ref true in
    
    List.iter (fun node1 ->
      match node1 with
      | Tree.Internal y1 ->
          List.iter (fun node2 ->
            match node2 with
            | Tree.Internal y2 when y1 <> y2 ->
                let sigma = compute_conditional_covariance 
                  params [y1; y2] tree in
                let det = Tensor.det sigma |> Tensor.get [] in
                if Float.abs det < 1e-10 then
                  valid := false
            | _ -> ()
          ) tree.nodes
      | _ -> ()
    ) tree.nodes;
    !valid

  let verify_convergence params_star params_t tree =
    verify_conditional_conservation params_t tree &&
    verify_conditional_expectations params_star params_t tree &&
    verify_linear_combination_property params_t tree &&
    verify_invertibility_conditions params_t tree
end

module FiniteSampleAnalysis = struct
  type complexity_bounds = {
    sample_complexity: int;
    iteration_complexity: int;
  }

  let compute_complexity_bounds n epsilon delta alpha beta =
    let c1 = 1. /. (alpha *. beta) in
    let c2 = 2. in
    let c3 = 1. /. alpha in
    let c4 = 2. in
    
    let m = int_of_float (
      c1 *. (float n ** c2) *. log (1. /. delta) /. (epsilon *. epsilon)
    ) in
    
    let t = int_of_float (
      c3 *. (float n ** c4) *. log (1. /. epsilon)
    ) in
    
    {sample_complexity = m; iteration_complexity = t}

  let verify_guarantees params true_params observations n_samples epsilon =
    let parameter_error = 
      Tensor.sub params.correlations true_params.correlations
      |> Tensor.abs
      |> Tensor.max []
      |> Tensor.get [] in
    
    let required_samples = 
      compute_complexity_bounds 
        (Tensor.size params.correlations 0) 
        epsilon 
        0.01  (* delta *)
        0.1   (* alpha *)
        0.1   (* beta *)
    in
    
    parameter_error < epsilon && 
    n_samples >= required_samples.sample_complexity

end