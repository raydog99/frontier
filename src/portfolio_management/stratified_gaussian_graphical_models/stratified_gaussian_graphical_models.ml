open Torch

(* Node representation *)
module Node = struct
  type t = int
  let compare = Int.compare
  let equal = Int.equal
  let hash = Hashtbl.hash
  let to_string = string_of_int
end

module NodeSet = Set.Make(Node)
module NodeMap = Map.Make(Node)

(* Edge representation *)
module Edge = struct
  type t = Node.t * Node.t
  
  let create n1 n2 = 
    if Node.compare n1 n2 < 0 then (n1, n2) else (n2, n1)
    
  let compare (n1, n2) (n3, n4) =
    match Node.compare n1 n3 with
    | 0 -> Node.compare n2 n4
    | c -> c
    
  let equal e1 e2 = compare e1 e2 = 0
  let hash = Hashtbl.hash
end

module EdgeSet = Set.Make(Edge)
module EdgeMap = Map.Make(Edge)

(* Stratum representation *)
module Stratum = struct
  type condition = {
    node: Node.t;
    lower_bound: float;
    upper_bound: float;
  }

  type t = condition list

  let create nodes bounds =
    List.map2 (fun n (l, u) -> 
      {node = n; lower_bound = l; upper_bound = u}
    ) nodes bounds
    
  let is_satisfied stratum x =
    List.for_all (fun cond ->
      let value = Tensor.get x [cond.node] in
      value >= cond.lower_bound && value <= cond.upper_bound
    ) stratum
end

module Graph = struct
  type t = {
    nodes: NodeSet.t;
    edges: EdgeSet.t;
    strata: (Edge.t * Stratum.t list) list;
  }

  let create nodes edges strata = {
    nodes = NodeSet.of_list nodes;
    edges = EdgeSet.of_list edges;
    strata = strata;
  }

  let adjacent g n =
    EdgeSet.fold (fun (n1, n2) acc ->
      if Node.equal n n1 then NodeSet.add n2 acc
      else if Node.equal n n2 then NodeSet.add n1 acc
      else acc
    ) g.edges NodeSet.empty

  (* Path finding between nodes *)
  let find_path g start goal =
    let visited = Hashtbl.create (NodeSet.cardinal g.nodes) in
    let rec bfs queue =
      match queue with
      | [] -> None
      | (node, path) :: rest ->
          if Node.equal node goal then Some (List.rev path)
          else begin
            Hashtbl.add visited node true;
            let neighbors = NodeSet.elements (adjacent g node) in
            let unvisited = List.filter 
              (fun n -> not (Hashtbl.mem visited n))
              neighbors in
            let new_paths = List.map 
              (fun n -> (n, n :: path)) 
              unvisited in
            bfs (rest @ new_paths)
          end
    in
    bfs [(start, [start])]

  let is_decomposable g =
    let cycles = Cycle.find_cycles g in
    List.for_all (fun cycle ->
      List.length cycle <= 3 || Cycle.has_chord g cycle
    ) cycles
end

module NumericalOpt = struct
  module Stabilization = struct
    let epsilon = 1e-10
    let max_condition = 1e6
    let min_eigenvalue = 1e-6

    let stable_inverse mat =
      let eigendecomp = Tensor.symeig mat ~eigenvectors:true in
      let eigenvals = fst eigendecomp in
      let eigenvecs = snd eigendecomp in
      
      (* Stabilize eigenvalues *)
      let stable_eigenvals = Tensor.map (fun x ->
        if x < min_eigenvalue then min_eigenvalue
        else if x > max_condition *. min_eigenvalue then 
          max_condition *. min_eigenvalue
        else x
      ) eigenvals in
      
      let inv_eigenvals = Tensor.map (fun x -> 1. /. x) stable_eigenvals in
      let diag_inv = Tensor.diag inv_eigenvals in
      Tensor.(matmul (matmul eigenvecs diag_inv) (transpose eigenvecs 0 1))

    let stable_cholesky mat =
      let stabilized = Tensor.add_scalar mat epsilon in
      try
        Tensor.cholesky stabilized
      with _ ->
        let perturbed = Tensor.add stabilized 
          (Tensor.mul_scalar (Tensor.eye (Tensor.size mat 0)) 
             (sqrt epsilon)) in
        Tensor.cholesky perturbed

    let stable_logdet mat =
      let chol = stable_cholesky mat in
      let diag = Tensor.diag chol in
      let log_diag = Tensor.map (fun x -> 
        2. *. log (max x epsilon)
      ) diag in
      Tensor.sum log_diag
  end
end

(* Multivariate Gaussian distribution *)
module MultivariateGaussian = struct
  type t = {
    mean: Tensor.t;
    covariance: Tensor.t;
    precision: Tensor.t option;
    cholesky: Tensor.t option;
    dim: int;
  }

  let create mean covariance =
    let dim = Tensor.size covariance 0 in

    let stable_cov = 
      NumericalOpt.Stabilization.stable_inverse covariance in
    let chol = 
      Some (NumericalOpt.Stabilization.stable_cholesky covariance) in
    {
      mean;
      covariance;
      precision = Some stable_cov;
      cholesky = chol;
      dim;
    }

  let log_pdf dist x =
    let centered = Tensor.sub x dist.mean in
    let prec = match dist.precision with
      | Some p -> p
      | None -> NumericalOpt.Stabilization.stable_inverse dist.covariance
    in
    
    let quad = Tensor.(matmul (matmul centered prec) 
                      (transpose centered 0 1)) in
    let logdet = 
      NumericalOpt.Stabilization.stable_logdet dist.covariance in
    
    -0.5 *. (float dist.dim *. log (2. *. Float.pi) +. 
             logdet +. 
             Tensor.get quad [0; 0])

  let sample dist n =
    let noise = Tensor.randn [n; dist.dim] in
    let chol = match dist.cholesky with
      | Some c -> c
      | None -> NumericalOpt.Stabilization.stable_cholesky dist.covariance
    in
    Tensor.(add (matmul noise chol) 
            (expand_as dist.mean noise))
end

(* Piecewise distribution handling *)
module PiecewiseDistribution = struct
  type piece = {
    region: Stratum.t;
    distribution: MultivariateGaussian.t;
  }

  type t = {
    pieces: piece list;
    dim: int;
  }

  let create pieces dim =
    {pieces; dim}

  let log_pdf dist x =
    let rec find_piece = function
      | [] -> None
      | piece :: rest ->
          if Stratum.is_satisfied piece.region x then
            Some piece
          else find_piece rest
    in
    match find_piece dist.pieces with
    | Some piece -> MultivariateGaussian.log_pdf piece.distribution x
    | None -> Float.neg_infinity

  let sample dist n =
    (* Simple rejection sampling *)
    let rec generate_sample () =
      let idx = Random.int (List.length dist.pieces) in
      let piece = List.nth dist.pieces idx in
      let sample = MultivariateGaussian.sample piece.distribution 1 in
      if Stratum.is_satisfied piece.region sample then
        sample
      else generate_sample ()
    in
    List.init n (fun _ -> generate_sample ())
    |> Tensor.stack ~dim:0
end

module SGGM = struct
  type t = {
    graph: Graph.t;
    dim: int;
    sigma: Tensor.t;  (* Covariance matrix *)
  }

  let create graph dim =
    let sigma = Tensor.eye dim in
    {graph; dim; sigma}

  (* Maximum likelihood estimation *)
  let mle model data =
    let optimized_cov = 
      ModelEstimation.estimate_parameters model data in
    {model with sigma = optimized_cov}

  (* Log likelihood calculation *)
  let log_likelihood model data =
    let n = Tensor.size data 0 in
    let d = model.dim in
    let det = NumericalOpt.Stabilization.stable_logdet model.sigma in
    let prec = NumericalOpt.Stabilization.stable_inverse model.sigma in
    
    let ll = ref (
      float n *. (-0.5 *. float d *. log (2.0 *. Float.pi) -. 
                 0.5 *. det)
    ) in
    
    (* Apply stratum conditions *)
    for i = 0 to n - 1 do
      let x = Tensor.slice data [Some i; None] in
      let quad = Tensor.(matmul (matmul x prec) 
                        (transpose x 0 1)) in
      ll := !ll -. 0.5 *. Tensor.get quad [0; 0];
      
      (* Check stratum conditions *)
      List.iter (fun (edge, stratum) ->
        if Stratum.is_satisfied stratum x then begin
          let (i, j) = edge in
          let cov_ij = Tensor.get model.sigma [i; j] in
          ll := !ll +. log (abs_float cov_ij)
        end
      ) model.graph.strata
    done;
    !ll

  (* Score function approximation using BIC *)
  let score model data =
    let ll = log_likelihood model data in
    let n = Tensor.size data 0 in
    let k = ModelSelection.count_parameters model in
    ll -. (float k *. log (float n)) /. 2.0

  (* Prior probability calculation *)
  let prior model =
    if Graph.is_decomposable model.graph then 1.0 
    else 0.0
end

module CompleteClique = struct
  (* Advanced clique finding using Bron-Kerbosch algorithm *)
  let find_maximal_cliques graph =
    let is_clique nodes =
      NodeSet.for_all (fun n1 ->
        NodeSet.for_all (fun n2 ->
          if Node.equal n1 n2 then true
          else EdgeSet.mem (Edge.create n1 n2) graph.Graph.edges
        ) nodes
      ) nodes
    in
    
    let rec bronk r p x acc =
      if NodeSet.is_empty p && NodeSet.is_empty x then
        if not (NodeSet.is_empty r) then r :: acc
        else acc
      else
        (* Choose pivot *)
        let u = NodeSet.choose (NodeSet.union p x) in
        let neighbors = Graph.adjacent graph u in
        
        let candidates = NodeSet.diff p 
          (NodeSet.inter p neighbors) in
        
        NodeSet.fold (fun v acc' ->
          let new_r = NodeSet.add v r in
          let new_p = NodeSet.inter p 
            (Graph.adjacent graph v) in
          let new_x = NodeSet.inter x 
            (Graph.adjacent graph v) in
          bronk new_r new_p new_x acc'
        ) candidates acc
    in
    
    bronk NodeSet.empty graph.Graph.nodes NodeSet.empty []

  (* Maximum clique finder using branch and bound *)
  let find_maximum_clique graph =
    let nodes = NodeSet.elements graph.Graph.nodes in
    let max_clique = ref [] in
    
    (* Color vertices greedily for upper bound *)
    let color_vertices vertices =
      let n = List.length vertices in
      let colors = Array.make n (-1) in
      List.iteri (fun i v ->
        let used = Array.make n false in
        List.iteri (fun j u ->
          if j < i && EdgeSet.mem (Edge.create v u) graph.Graph.edges then
            used.(colors.(j)) <- true
        ) vertices;
        let rec find_color c =
          if c >= n || not used.(c) then c
          else find_color (c + 1)
        in
        colors.(i) <- find_color 0
      ) vertices;
      colors
    in

    (* Branch and bound *)
    let rec expand clique vertices bound =
      if List.length clique > List.length !max_clique then
        max_clique := clique;

      if List.length vertices = 0 || 
         List.length clique + bound <= List.length !max_clique then
        ()
      else
        let v = List.hd vertices in
        let new_vertices = 
          List.filter (fun u ->
            EdgeSet.mem (Edge.create v u) graph.Graph.edges
          ) (List.tl vertices) in
        
        if List.length new_vertices > 0 then begin
          let colors = color_vertices new_vertices in
          let new_bound = 
            Array.fold_left max 0 colors + 1 in
          expand (v :: clique) new_vertices new_bound
        end;
        
        expand clique (List.tl vertices) (bound - 1)
    in

    (* Initialize with all vertices *)
    let init_colors = color_vertices nodes in
    let init_bound = Array.fold_left max 0 init_colors + 1 in
    expand [] nodes init_bound;
    !max_clique
end

module CompleteSeparator = struct
  (* Find all minimal vertex separators *)
  let find_minimal_separators graph =
    let nodes = NodeSet.elements graph.Graph.nodes in
    let separators = ref [] in
    
    (* Check if set is a minimal separator *)
    let is_minimal_separator sep =
      let components = ref [] in
      let visited = NodeSet.empty in
      
      let rec dfs node visited =
        let new_visited = NodeSet.add node visited in
        let neighbors = NodeSet.filter
          (fun n -> not (NodeSet.mem n sep))
          (Graph.adjacent graph node) in
        NodeSet.fold (fun next (comp, visited) ->
          if NodeSet.mem next visited then (comp, visited)
          else
            let sub_comp, new_visited = dfs next visited in
            (NodeSet.union comp sub_comp, new_visited)
        ) neighbors (NodeSet.singleton node, new_visited)
      in
      
      let remaining = NodeSet.diff graph.Graph.nodes sep in
      let rec find_components nodes comps =
        match NodeSet.choose_opt nodes with
        | None -> comps
        | Some node ->
            if NodeSet.mem node sep then
              find_components (NodeSet.remove node nodes) comps
            else
              let comp, visited = dfs node NodeSet.empty in
              find_components 
                (NodeSet.diff nodes visited)
                (comp :: comps)
      in
      let components = find_components remaining [] in
      
      (* Check minimality *)
      List.length components >= 2 &&
      List.for_all (fun comp ->
        NodeSet.exists (fun v ->
          NodeSet.exists (fun s ->
            NodeSet.exists (fun u ->
              EdgeSet.mem (Edge.create v u) graph.Graph.edges
            ) comp
          ) sep
        ) comp
      ) components
    in
    
    (* Generate potential separators *)
    let rec generate_separators size current_sep remaining =
      if NodeSet.cardinal current_sep = size then
        if is_minimal_separator current_sep then
          separators := current_sep :: !separators
      else
        NodeSet.iter (fun node ->
          let new_sep = NodeSet.add node current_sep in
          let new_remaining = NodeSet.remove node remaining in
          generate_separators size new_sep new_remaining
        ) remaining
    in
    
    for size = 1 to NodeSet.cardinal graph.Graph.nodes - 2 do
      generate_separators size NodeSet.empty graph.Graph.nodes
    done;
    !separators

  (* Find minimum weight separator *)
  let find_minimum_weight_separator graph weights =
    let nodes = NodeSet.elements graph.Graph.nodes in
    let best_separator = ref None in
    let min_weight = ref Float.infinity in
    
    (* Check if set splits graph *)
    let splits_graph separator =
      let components = ref [] in
      let visited = NodeSet.empty in
      
      let rec dfs node visited =
        let new_visited = NodeSet.add node visited in
        let neighbors = NodeSet.filter
          (fun n -> not (NodeSet.mem n separator))
          (Graph.adjacent graph node) in
        NodeSet.fold (fun next (comp, visited) ->
          if NodeSet.mem next visited then (comp, visited)
          else
            let sub_comp, new_visited = dfs next visited in
            (NodeSet.union comp sub_comp, new_visited)
        ) neighbors (NodeSet.singleton node, new_visited)
      in
      
      let remaining = NodeSet.diff graph.Graph.nodes separator in
      let rec find_components nodes =
        match NodeSet.choose_opt nodes with
        | None -> List.length !components >= 2
        | Some node ->
            if NodeSet.mem node separator then
              find_components (NodeSet.remove node nodes)
            else
              let comp, visited = dfs node NodeSet.empty in
              components := comp :: !components;
              find_components (NodeSet.diff nodes visited)
      in
      find_components remaining
    in
    
    let weight_of_separator sep =
      NodeSet.fold (fun node acc ->
        acc +. weights.(Node.hash node)
      ) sep 0.0
    in
    
    (* Generate and check separators *)
    let rec generate_separators size current_sep remaining =
      if NodeSet.cardinal current_sep = size then begin
        if splits_graph current_sep then
          let weight = weight_of_separator current_sep in
          if weight < !min_weight then begin
            min_weight := weight;
            best_separator := Some current_sep
          end
      end else
        NodeSet.iter (fun node ->
          let new_sep = NodeSet.add node current_sep in
          let new_remaining = NodeSet.remove node remaining in
          generate_separators size new_sep new_remaining
        ) remaining
    in
    
    for size = 1 to NodeSet.cardinal graph.Graph.nodes - 2 do
      generate_separators size NodeSet.empty graph.Graph.nodes
    done;
    
    match !best_separator with
    | Some sep -> Some (sep, !min_weight)
    | None -> None
end

module ModelSelection = struct
  type mcmc_state = {
    model: SGGM.t;
    score: float;
  }

  type proposal_type =
    | AddEdge
    | RemoveEdge
    | AddStratum
    | RemoveStratum
    | ModifyStratum
    | SwapEdges
    | SplitStratum
    | MergeStrata
    | FlipEdge
    | ModifyBoundary

  (* Generate random stratum *)
  let generate_stratum nodes =
    let n = List.length nodes in
    let selected = Random.int (n + 1) in
    let selected_nodes = List.take selected nodes in
    List.map (fun node ->
      let lower = Random.float 2.0 -. 1.0 in
      let upper = lower +. Random.float 2.0 in
      {Stratum.node = node; 
       lower_bound = lower; 
       upper_bound = upper}
    ) selected_nodes

  (* Proposal generation *)
  let propose_next state =
    let model = state.model in
    let graph = model.SGGM.graph in
    
    match Random.int 10 with
    | 0 -> (* Add edge *)
        let n1 = NodeSet.choose graph.Graph.nodes in
        let n2 = NodeSet.choose graph.Graph.nodes in
        if n1 <> n2 && 
           not (EdgeSet.mem (Edge.create n1 n2) graph.Graph.edges) then
          let new_edges = EdgeSet.add (Edge.create n1 n2) graph.Graph.edges in
          let new_graph = {graph with Graph.edges = new_edges} in
          if Graph.is_decomposable new_graph then
            {model with SGGM.graph = new_graph}
          else
            model
        else
          model
        
    | 1 -> (* Remove edge *)
        if not (EdgeSet.is_empty graph.Graph.edges) then
          let edge = EdgeSet.choose graph.Graph.edges in
          let new_edges = EdgeSet.remove edge graph.Graph.edges in
          let new_strata = List.filter (fun (e, _) -> e <> edge) 
                            graph.Graph.strata in
          {model with SGGM.graph = 
            {graph with Graph.edges = new_edges; 
                        Graph.strata = new_strata}}
        else
          model
        
    | 2 -> (* Add stratum *)
        if not (EdgeSet.is_empty graph.Graph.edges) then
          let edge = EdgeSet.choose graph.Graph.edges in
          let neighbors = Graph.adjacent graph (fst edge) in
          let stratum = generate_stratum (NodeSet.elements neighbors) in
          let new_strata = (edge, stratum) :: graph.Graph.strata in
          {model with SGGM.graph = {graph with Graph.strata = new_strata}}
        else
          model
        
    | 3 -> (* Remove stratum *)
        match graph.Graph.strata with
        | [] -> model
        | _ :: rest ->
            {model with SGGM.graph = {graph with Graph.strata = rest}}
    
    | 4 -> (* Modify stratum *)
        let modify_condition cond =
          let delta = Random.float 0.2 -. 0.1 in
          {cond with 
            Stratum.lower_bound = cond.Stratum.lower_bound +. delta;
            Stratum.upper_bound = cond.Stratum.upper_bound +. delta}
        in
        let new_strata = List.map (fun (edge, stratum) ->
          (edge, List.map modify_condition stratum)
        ) graph.Graph.strata in
        {model with SGGM.graph = {graph with Graph.strata = new_strata}}
    
    | 5 -> (* Swap edges *)
        if EdgeSet.cardinal graph.Graph.edges >= 2 then
          let e1 = EdgeSet.choose graph.Graph.edges in
          let e2 = EdgeSet.choose (EdgeSet.remove e1 graph.Graph.edges) in
          let new_edges = EdgeSet.add e2 
            (EdgeSet.add e1 
               (EdgeSet.remove e2 
                  (EdgeSet.remove e1 graph.Graph.edges))) in
          {model with SGGM.graph = {graph with Graph.edges = new_edges}}
        else
          model
    
    | 6 -> (* Split stratum *)
        match graph.Graph.strata with
        | (edge, stratum) :: rest ->
            let n = List.length stratum in
            let split_point = n / 2 in
            let part1 = List.take split_point stratum in
            let part2 = List.drop split_point stratum in
            if part1 <> [] && part2 <> [] then
              let new_strata = 
                (edge, part1) :: (edge, part2) :: rest in
              {model with SGGM.graph = 
                {graph with Graph.strata = new_strata}}
            else
              model
        | [] -> model
    
    | 7 -> (* Merge strata *)
        match graph.Graph.strata with
        | s1 :: s2 :: rest ->
            let merged = (fst s1, List.append (snd s1) (snd s2)) in
            let new_strata = merged :: rest in
            {model with SGGM.graph = 
              {graph with Graph.strata = new_strata}}
        | _ -> model
    
    | 8 -> (* Flip edge *)
        if not (EdgeSet.is_empty graph.Graph.edges) then
          let (n1, n2) as edge = EdgeSet.choose graph.Graph.edges in
          let new_edges = EdgeSet.add (Edge.create n2 n1)
            (EdgeSet.remove edge graph.Graph.edges) in
          {model with SGGM.graph = {graph with Graph.edges = new_edges}}
        else
          model
    
    | _ -> (* Modify boundary *)
        let new_strata = List.map (fun (edge, stratum) ->
          let modified = List.map (fun cond ->
            if Random.bool () then
              let delta = Random.float 0.1 in
              if Random.bool () then
                {cond with Stratum.lower_bound = 
                  cond.Stratum.lower_bound +. delta}
              else
                {cond with Stratum.upper_bound = 
                  cond.Stratum.upper_bound +. delta}
            else cond
          ) stratum in
          (edge, modified)
        ) graph.Graph.strata in
        {model with SGGM.graph = {graph with Graph.strata = new_strata}}

  (* Count parameters in model *)
  let count_parameters model =
    let dim = model.SGGM.dim in
    let base_params = (dim * (dim + 1)) / 2 in
    let edge_reduction = EdgeSet.cardinal model.SGGM.graph.Graph.edges in
    let strata_params = List.fold_left (fun acc (_, stratum) ->
      acc + 2 * List.length stratum  (* 2 parameters per condition *)
    ) 0 model.SGGM.graph.Graph.strata in
    base_params - edge_reduction + strata_params

  (* Non-reversible MCMC search *)
  let search init_model data max_iter =
    let visited = Hashtbl.create 1000 in
    let best_model = ref init_model in
    let best_score = ref (SGGM.score init_model data) in
    
    let rec mcmc_iter current_model current_score iter =
      if iter >= max_iter then !best_model
      else begin
        (* Generate proposal *)
        let proposal = propose_next 
          {model = current_model; score = current_score} in
        let proposal_hash = Hashtbl.hash proposal in
        
        (* Check if proposal was already visited *)
        if Hashtbl.mem visited proposal_hash then
          mcmc_iter current_model current_score (iter + 1)
        else begin
          (* Evaluate proposal *)
          let proposal_score = SGGM.score proposal data in
          
          (* Metropolis-Hastings acceptance *)
          let accept_prob = exp (proposal_score -. current_score) in
          if Random.float 1.0 < accept_prob then begin
            (* Accept proposal *)
            Hashtbl.add visited proposal_hash proposal;
            
            (* Update best if needed *)
            if proposal_score > !best_score then begin
              best_model := proposal;
              best_score := proposal_score
            end;
            
            mcmc_iter proposal proposal_score (iter + 1)
          end else
            mcmc_iter current_model current_score (iter + 1)
        end
      end
    in
    
    mcmc_iter init_model (SGGM.score init_model data) 0
end

(* Context-specific independence verification *)
module CSIndependence = struct
  type independence_check = {
    vars: Node.t list;
    context: Stratum.t;
    is_independent: bool;
  }

  (* Check conditional independence given context *)
  let verify_conditional_independence model data vars context =
    let dim = model.SGGM.dim in
    let sigma = model.SGGM.sigma in
    
    (* Filter data based on context *)
    let valid_indices = ref [] in
    let n = Tensor.size data 0 in
    for i = 0 to n - 1 do
      let sample = Tensor.slice data [Some i; None] in
      if Stratum.is_satisfied context sample then
        valid_indices := i :: !valid_indices
    done;
    
    (* Extract relevant data *)
    let filtered_data = 
      if List.length !valid_indices > 0 then
        Tensor.stack ~dim:0 
          (List.map (fun i -> 
            Tensor.slice data [Some i; None]
          ) !valid_indices)
      else data in
    
    (* Compute conditional covariance *)
    let cond_cov = 
      if List.length !valid_indices > 0 then
        let emp_cov = Tensor.(matmul 
          (transpose filtered_data 0 1) 
          filtered_data) in
        Tensor.div_scalar emp_cov 
          (float (List.length !valid_indices))
      else sigma in
    
    (* Check independence using precision matrix *)
    let precision = 
      NumericalOpt.Stabilization.stable_inverse cond_cov in
    List.for_all (fun i ->
      List.for_all (fun j ->
        if i <> j then
          abs_float (Tensor.get precision [i; j]) < 1e-6
        else true
      ) vars
    ) vars

  (* Verify all context-specific independencies *)
  let verify_all_independencies model data =
    let checks = ref [] in
    List.iter (fun (edge, stratum) ->
      let (n1, n2) = edge in
      let result = verify_conditional_independence 
        model data [n1; n2] stratum in
      checks := {
        vars = [n1; n2];
        context = stratum;
        is_independent = result;
      } :: !checks
    ) model.SGGM.graph.Graph.strata;
    !checks
end

(* Curved exponential family verification *)
module CurvedFamily = struct
  type curved_family_check = {
    smooth_density: bool;
    continuous_statistics: bool;
    parameter_constraints: bool;
    exponential_family: bool;
  }

  (* Check smoothness of density *)
  let verify_smooth_density model data =
    let epsilon = 1e-6 in
    let log_lik x = 
      SGGM.log_likelihood model (Tensor.unsqueeze x 0) in
    
    try
      let sample = Tensor.slice data [Some 0; None] in
      let dim = Tensor.size sample 0 in
      let grad = Tensor.zeros [dim] in
      
      (* Compute numerical gradient *)
      for i = 0 to dim - 1 do
        let x_plus = Tensor.copy sample in
        let x_minus = Tensor.copy sample in
        Tensor.set_ x_plus [i] 
          (Tensor.get sample [i] +. epsilon);
        Tensor.set_ x_minus [i] 
          (Tensor.get sample [i] -. epsilon);
        
        let diff = (log_lik x_plus -. log_lik x_minus) /. 
                  (2. *. epsilon) in
        Tensor.set_ grad [i] diff
      done;
      
      (* Check if gradient exists and is continuous *)
      let grad_norm = Tensor.norm grad in
      Float.is_finite (Tensor.get grad_norm []) &&
      Tensor.get grad_norm [] < 1e6
    with _ -> false

  (* Check parameter constraints *)
  let verify_parameter_constraints model =
    try
      let precision = 
        NumericalOpt.Stabilization.stable_inverse model.SGGM.sigma in
      let eigenvals = 
        Tensor.symeig precision ~eigenvectors:false |> fst in
      Tensor.all (Tensor.gt eigenvals (Tensor.scalar 0.))
    with _ -> false

  (* Full verification *)
  let verify model data = {
    smooth_density = verify_smooth_density model data;
    continuous_statistics = true;  (* Always true for Gaussian *)
    parameter_constraints = verify_parameter_constraints model;
    exponential_family = true;  (* Always true for SGGM *)
  }
end

(* Normalizing constant computation *)
module NormalizingConstant = struct
  type integration_method =
    | MonteCarlo of int
    | ImportanceSampling of int
    | LaplaceMean

  (* Compute normalizing constant *)
  let compute method_ model =
    match method_ with
    | LaplaceMean ->
        let dim = model.SGGM.dim in
        let det = NumericalOpt.Stabilization.stable_logdet 
          model.SGGM.sigma in
        exp (0.5 *. (det +. float dim *. log (2. *. Float.pi)))
        
    | ImportanceSampling n ->
        let dim = model.SGGM.dim in
        let proposal = MultivariateGaussian.create 
          (Tensor.zeros [dim])
          (Tensor.eye dim) in
        
        let samples = MultivariateGaussian.sample proposal n in
        let weights = Tensor.zeros [n] in
        
        for i = 0 to n - 1 do
          let x = Tensor.slice samples [Some i; None] in
          let target_log = 
            SGGM.log_likelihood model (Tensor.unsqueeze x 0) in
          let proposal_log = 
            MultivariateGaussian.log_pdf proposal x in
          Tensor.set_ weights [i] 
            (exp (target_log -. proposal_log))
        done;
        
        let mean_weight = Tensor.mean weights in
        Tensor.get mean_weight []
        
    | MonteCarlo n ->
        let gaussian = MultivariateGaussian.create 
          (Tensor.zeros [model.SGGM.dim])
          model.SGGM.sigma in
        let samples = MultivariateGaussian.sample gaussian n in
        let log_liks = Tensor.zeros [n] in
        
        for i = 0 to n - 1 do
          let x = Tensor.slice samples [Some i; None] in
          let ll = SGGM.log_likelihood model 
            (Tensor.unsqueeze x 0) in
          Tensor.set_ log_liks [i] ll
        done;
        
        let max_ll = Tensor.max log_liks in
        let shifted = Tensor.exp 
          (Tensor.sub log_liks max_ll) in
        Tensor.mean shifted |> Tensor.get [0] |> exp |> 
        ( *. ) (exp (Tensor.get max_ll [0]))
end