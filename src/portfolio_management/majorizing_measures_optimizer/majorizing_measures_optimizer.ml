open Torch

(* Metric space represented by pairwise distances *)
type metric_space = {
    points: tensor;
    distances: tensor; (* n x n distance matrix *)
    n_points: int;
}

(* Measure represented as weights over points *)
type measure = {
    weights: tensor;
    support: tensor; (* boolean mask for support *)
}

(* Chaining functional from (0,1] -> R+ *)
type chaining_functional = {
    f: float -> float;           (* density function *)
    f_inv: float -> float;       (* inverse of F(s) = int_s^inf f(x)dx *)
    is_log_concave: bool;        (* flag for log-concave type *)
}

(* Ball in metric space *)
type ball = {
    center: int;   (* index of center point *)
    radius: float;
    metric: metric_space;
}

(* Tree node types *)
type 'a tree_node = {
    data: 'a;
    children: 'a tree_node list;
}

(* Complete packing tree node *)
type complete_packing_node = {
    vertices: tensor;           (* Set of vertices *)
    chi: int;                  (* χ(V) label *)
    center: int;               (* Center point *)
    radius: float;             (* Node radius *)
    diameter: float;           (* Cached diameter *)
    separation: float;         (* Minimum separation from siblings *)
    parent_radius: float;      (* Parent node radius *)
    children: complete_packing_node list;
    weight: float;             (* Node weight for balancing *)
}

(* Complete labelled net node *)
type complete_node = {
    vertices: tensor;     (* Set of vertices *)
    m_label: int;        (* m(V) label *)
    sigma: int;          (* σ(V) ordering *)
    center: int;         (* Center point *)
    diameter: float;     (* Cached diameter *)
    children: complete_node list;
}

(* Core measure operations *)
module Measure = struct
  (* Create measure from weights *)
  let create weights =
    let support = Tensor.(weights > scalar 0.) in
    {weights; support}

  (* Get measure of a subset *)
  let measure_subset measure indices =
    Tensor.(sum (weights measure * indices))

  (* Get measure of a ball *)
  let measure_ball measure ball =
    let in_ball = Tensor.(
      distances ball.metric <= scalar ball.radius
      |> get_index1 ball.center
    ) in
    measure_subset measure in_ball

  (* Check if measure is normalized *)
  let is_normalized measure =
    Tensor.(abs (sum measure.weights - scalar 1.) < scalar 1e-6)

  (* Normalize measure *)
  let normalize measure =
    let total = Tensor.(sum measure.weights |> get_all1) in
    if abs_float (total -. 1.) < 1e-6 then measure
    else
      {measure with weights = Tensor.(measure.weights / scalar total)}
end

(* Ball operations *)
module Ball = struct
  (* Create ball from center and radius *)
  let create metric center radius =
    {center; radius; metric}

  (* Check if point is in ball *)
  let contains ball point =
    Tensor.(get2 ball.metric.distances point ball.center <= scalar ball.radius)

  (* Get points in ball as boolean mask *)
  let get_points ball =
    Tensor.(distances ball.metric <= scalar ball.radius
           |> get_index1 ball.center)

  (* Compute diameter of a subset *)
  let diameter metric points =
    let mask = Tensor.(points.unsqueeze(-1) * points) in
    Tensor.(masked_select metric.distances mask |> max)
end

(* Basic metric space operations *)
module Metric = struct
  (* Create metric space from distance matrix *)
  let create distances =
    let n = Tensor.size distances 0 in
    let points = Tensor.arange ~start:0 ~end_:n ~options:(Device CPU) in
    {points; distances; n_points = n}

  (* Get distance between points *)
  let distance metric i j =
    Tensor.get2 metric.distances i j

  (* Find closest point to a given point *)
  let closest_point metric point mask =
    let dists = Tensor.get_index1 metric.distances point in
    let masked_dists = Tensor.(dists * mask + (scalar Float.max_float) * (~ mask)) in
    Tensor.argmin masked_dists ~dim:0 
end

(* Chaining functional *)
module ChainingFunctional = struct
  (* Create standard Gaussian chaining functional *)
  let create_gaussian () = {
    f = (fun x -> exp (-0.5 *. x *. x));
    f_inv = (fun p -> sqrt (-2. *. log p));
    is_log_concave = true;
  }

  (* Evaluate h(p) = F^{-1}(p) *)
  let evaluate h p =
    if p <= 0. || p > 1. then invalid_arg "p must be in (0,1]"
    else h.f_inv p

  (* Check sub-multiplicative property *)
  let check_submultiplicative h a b =
    if a <= 0. || a > 1. || b <= 0. || b > 1. then
      invalid_arg "a,b must be in (0,1]"
    else
      let hab = evaluate h (a *. b) in
      let ha = evaluate h a in
      let hb = evaluate h b in
      hab <= ha +. hb

  (* Verify bounds *)
  let check_bounds h a =
    if a <= 0. || a > 1. then
      invalid_arg "a must be in (0,1]"
    else
      let ha = evaluate h a in
      let log_bound = -. log a in
      let deriv_bound = -1. in
      ha <= log_bound && 
      a *. (h.f (evaluate h a)) >= deriv_bound
end

(* Enhanced chaining functional *)
module EnhancedChaining = struct
  (* Complete F(s) with proper bounds *)
  let compute_F f s =
    let rec integrate acc x steps =
      if steps > 1000 || x > s +. 100. || acc > 1. then acc
      else
        let dx = min 0.001 (1. /. (float steps +. 1.)) in
        let fx = f x in
        integrate (acc +. fx *. dx) (x +. dx) (steps + 1)
    in
    min 1. (integrate 0. s 0)

  (* Create distribution with proper validation *)
  let create_distribution f =
    (* Verify function is non-decreasing *)
    let verify_non_decreasing f =
      let points = List.init 1000 (fun i -> float_of_int i /. 100.) in
      let rec check = function
        | [] | [_] -> true
        | x::y::rest -> f x <= f y && check (y::rest)
      in
      check points
    in

    (* Verify log-concavity *)
    let verify_log_concave f =
      let points = List.init 1000 (fun i -> float_of_int i /. 100.) in
      let log_f x = log (max 1e-10 (f x)) in
      let rec check = function
        | [] | [_] | [_;_] -> true
        | x::y::z::rest ->
            let d1 = (log_f y -. log_f x) /. (y -. x) in
            let d2 = (log_f z -. log_f y) /. (z -. y) in
            d1 >= d2 && check (y::z::rest)
      in
      check points
    in

    if not (verify_non_decreasing f) then
      invalid_arg "Density function must be non-decreasing";
    
    let is_log_concave = verify_log_concave f in
    let F = compute_F f in
    
    (* Binary search for F inverse *)
    let F_inv p =
      if p <= 0. || p > 1. then invalid_arg "p must be in (0,1]";
      let rec search left right =
        if right -. left < 1e-6 then right
        else
          let mid = (left +. right) /. 2. in
          if F mid > p then
            search left mid
          else
            search mid right
      in
      search 0. 100.
    in
    
    {f; F; F_inv; log_concave = is_log_concave; scale = 1.}

  (* Implement rescaling symmetry *)
  let rescale dist beta =
    if beta <= 0. then invalid_arg "beta must be positive";
    let f x = beta *. dist.f (beta *. x) in
    let F s = dist.F (beta *. s) in
    let F_inv p = dist.F_inv p /. beta in
    {f; F; F_inv; log_concave = dist.log_concave; scale = beta *. dist.scale}

  (* Complete h(p) verification *)
  let verify_h h p =
    let deriv_check p =
      let eps = 1e-6 in
      let h1 = h.f_inv p in
      let h2 = h.f_inv (p +. eps) in
      let deriv = (h2 -. h1) /. eps in
      p *. deriv >= -1. && p *. deriv <= 0.
    in
    p > 0. && p <= 1. &&
    h.f_inv p >= 0. &&
    h.f_inv p <= -. log p &&
    deriv_check p
end

(* Enhanced measure operations *)
module EnhancedMeasure = struct
  (* Cache for ball measures *)
  type measure_cache = {
    balls: (int * float, float) Hashtbl.t;  (* (center, radius) -> measure *)
    subsets: (string, float) Hashtbl.t;     (* vertex set key -> measure *)
  }

  (* Create empty cache *)
  let create_cache () = {
    balls = Hashtbl.create 1000;
    subsets = Hashtbl.create 1000;
  }

  (* Cached measure of ball *)
  let cached_ball_measure cache measure ball =
    let key = (ball.center, ball.radius) in
    match Hashtbl.find_opt cache.balls key with
    | Some v -> v
    | None ->
        let v = Measure.measure_ball measure ball |> Tensor.get_all1 in
        Hashtbl.add cache.balls key v;
        v

  (* Cached measure of subset *)
  let cached_subset_measure cache measure vertices =
    let key = Tensor.to_string vertices in
    match Hashtbl.find_opt cache.subsets key with
    | Some v -> v
    | None ->
        let v = Measure.measure_subset measure vertices |> Tensor.get_all1 in
        Hashtbl.add cache.subsets key v;
        v

  (* Verify measure properties *)
  let verify_measure measure =
    let non_negative = Tensor.(all (weights measure >= scalar 0.)) in
    let total = Tensor.(sum measure.weights |> get_all1) in
    let normalized = abs_float (total -. 1.) < 1e-6 in
    non_negative && normalized
end

(* Graph theory components *)
module Graph = struct
  (* Bipartite graph representation *)
  type bipartite_graph = {
    x1: tensor;  (* First vertex set *)
    x2: tensor;  (* Second vertex set *)
    edges: tensor; (* Adjacency matrix *)
    weights_x1: measure;  (* Weights on X1 *)
    weights_x2: measure;  (* Weights on X2 *)
    metric: metric_space;
  }

  (* Edge set representation *)
  type edge_set = {
    vertices: tensor * tensor;  (* Pairs of connected vertices *)
    weights: tensor;  (* Edge weights *)
  }

  (* Create bipartite graph from metric space and radius *)
  let create_bipartite metric radius mu nu =
    let n = metric.n_points in
    let edges = Tensor.(distances metric <= scalar radius) in
    {
      x1 = Tensor.arange ~start:0 ~end_:n ~options:(Device CPU);
      x2 = Tensor.arange ~start:0 ~end_:n ~options:(Device CPU);
      edges;
      weights_x1 = mu;
      weights_x2 = nu;
      metric;
    }

  (* Get neighbors of a vertex set *)
  let neighbors graph vertices side =
    let adj = match side with
      | `Left -> graph.edges
      | `Right -> Tensor.transpose graph.edges ~dim0:0 ~dim1:1
    in
    Tensor.(sum (vertices.unsqueeze(-1) * adj) > scalar 0.)

  (* Get incident edges on vertex *)
  let incident_edges graph v side =
    match side with
    | `Left -> Tensor.get_index1 graph.edges v
    | `Right -> Tensor.get_index1 (Tensor.transpose graph.edges ~dim0:0 ~dim1:1) v

  (* Compute 2-hop neighbors *)
  let neighbors_2hop graph vertices =
    neighbors graph (neighbors graph vertices `Right) `Left

  (* Edge incident set *)
  let incident_edges_set graph v =
    match Tensor.size graph.edges 0 with
    | n -> 
        let edges = Tensor.zeros [n] in
        for i = 0 to n-1 do
          if Tensor.get_all1 (Tensor.get2 graph.edges v i) then
            Tensor.set1 edges i (Tensor.scalar 1.)
        done;
        edges

  (* Set operations *)
  let union t1 t2 = Tensor.(t1 + t2 > scalar 0.)
  let intersection t1 t2 = Tensor.(t1 * t2)
  let difference t1 t2 = Tensor.(t1 * (scalar 1. - t2))
end

(* Principal sequences *)
module PrincipalSequence = struct
  (* Principal sequence element *)
  type sequence_element = {
    set: tensor;          (* Set Si *)
    beta: float;         (* βi value *)
    neighbors: tensor;    (* N(Si) \ N(Si-1) *)
    weight_ratio: float; (* μ(Si \ Si-1) / ν(N(Si) \ N(Si-1)) *)
  }

  type principal_sequence = {
    elements: sequence_element list;
    metric: metric_space;
    base_measure: measure;
  }

  (* Check matchability condition *)
  let verify_matchable set beta graph mu nu =
    let neighbors = neighbors graph set `Right in
    let mu_set = Measure.measure_subset mu set in
    let nu_neighbors = Measure.measure_subset nu neighbors in
    Tensor.get_all1 (Tensor.(mu_set * scalar beta <= nu_neighbors))

  (* Find optimal beta *)
  let find_optimal_beta graph mu nu set =
    let neighbors = neighbors graph set `Right in
    let mu_set = Measure.measure_subset mu set in
    let nu_neighbors = Measure.measure_subset nu neighbors in
    Tensor.get_all1 (Tensor.(nu_neighbors / mu_set))

  (* Find minimal unmatchable set *)
  let find_minimal_unmatchable graph mu nu beta current_set =
    let n = Tensor.size graph.x1 0 in
    let best_set = ref current_set in
    let best_size = ref (Tensor.get_all1 (Tensor.sum current_set)) in
    
    let rec search set size =
      if verify_matchable set beta graph mu nu then ()
      else if Tensor.get_all1 (Tensor.sum set) < !best_size then begin
        best_set := set;
        best_size := size
      end else
        for i = 0 to n-1 do
          if not (Tensor.get_all1 (Tensor.get1 set i)) then
            let new_set = Tensor.(set + (ones [n] |> get1) i) in
            search new_set (size +. 1.)
        done
    in
    
    search current_set (Tensor.get_all1 (Tensor.sum current_set));
    !best_set

  (* Compute complete principal sequence *)
  let compute_sequence graph mu nu =
    let n = Tensor.size graph.x1 0 in
    
    let rec build_sequence current_set elements =
      if Tensor.(sum current_set == scalar (float n)) then
        List.rev elements
      else
        let beta = find_optimal_beta graph mu nu current_set in
        let min_set = find_minimal_unmatchable graph mu nu beta current_set in
        let neighbors = difference 
          (neighbors graph min_set `Right)
          (neighbors graph current_set `Right) in
        
        let weight_ratio = 
          let set_diff = difference min_set current_set in
          let mu_diff = Measure.measure_subset mu set_diff in
          let nu_neighbors = Measure.measure_subset nu neighbors in
          Tensor.get_all1 (Tensor.(mu_diff / nu_neighbors))
        in
        
        let element = {
          set = min_set;
          beta;
          neighbors;
          weight_ratio;
        } in
        
        build_sequence min_set (element :: elements)
    in
    
    let elements = build_sequence (Tensor.zeros [n]) [] in
    {elements; metric = graph.metric; base_measure = mu}
end

(* Complete Labelled Nets *)
module CompleteLabelledNet = struct
  type complete_net = {
    root: complete_node;
    metric: metric_space;
    alpha: float;        (* α parameter *)
    measure: measure;    (* Associated measure *)
  }

  (* Create complete labelled net *)
  let create metric measure alpha =
    if alpha <= 0. || alpha > 0.1 then
      invalid_arg "alpha must be in (0,1/10]";
    
    let diam = Ball.diameter metric (Tensor.ones [metric.n_points]) in
    let root = {
      vertices = Tensor.ones [metric.n_points];
      m_label = 0;
      sigma = 1;
      center = 0;
      diameter = diam;
      children = []
    } in
    
    {root; metric; alpha; measure}

  (* Proper value computation *)
  let compute_value net h =
    let rec compute_path_value node =
      match node.children with
      | [] -> []
      | cs ->
          List.concat_map (fun child ->
            let edge_value = 
              net.alpha ** float node.m_label *. 
              node.diameter *.
              ChainingFunctional.evaluate h (1. /. float child.sigma)
            in
            edge_value :: compute_path_value child
          ) cs
    in
    
    let all_paths = compute_path_value net.root in
    List.fold_left max 0. all_paths

  (* Optimal partition finding *)
  let find_optimal_partition node net =
    let radius = 0.5 *. net.alpha ** float (node.m_label + 2) *. net.root.diameter in
    let excluded = ref (Tensor.zeros [net.metric.n_points]) in
    let partitions = ref [] in
    
    while Tensor.(sum !excluded < scalar (float net.metric.n_points)) do
      let center = ref (-1) in
      let max_measure = ref Float.neg_infinity in
      
      for i = 0 to net.metric.n_points - 1 do
        if not (Tensor.get_all1 (Tensor.get1 !excluded i)) then
          let ball = Ball.create net.metric i radius in
          let m = Measure.measure_ball net.measure ball in
          if Tensor.get_all1 m > !max_measure then begin
            center := i;
            max_measure := Tensor.get_all1 m
          end
      done;
      
      if !center >= 0 then begin
        let ball = Ball.create net.metric !center radius in
        let vertices = Ball.get_points ball in
        excluded := Tensor.(!excluded + vertices);
        partitions := (!center, vertices) :: !partitions
      end
    done;
    
    List.rev !partitions

  (* Build complete tree recursively *)
  let rec build_tree_recursive parent net =
    let partitions = find_optimal_partition parent net in
    
    let children = List.mapi (fun i (center, vertices) ->
      let child = {
        vertices;
        m_label = parent.m_label + 1;
        sigma = i + 1;
        center;
        diameter = Ball.diameter net.metric vertices;
        children = []
      } in
      
      if Tensor.(sum vertices == scalar 1.) then child
      else
        let child_net = {net with root = child} in
        let subtree = build_tree_recursive child child_net in
        {subtree with m_label = child.m_label; sigma = child.sigma}
    ) partitions in
    
    {parent with children}

  (* Verify net properties *)
  let verify_net net =
    let rec verify_node parent =
      (* Verify diameter condition *)
      let diam_ok = parent.diameter <= 
        net.alpha ** float parent.m_label *. net.root.diameter in
      
      (* Verify label conditions *)
      let labels_ok = List.for_all (fun child ->
        child.m_label >= parent.m_label + 1
      ) parent.children in
      
      (* Verify node separation *)
      let separation_ok =
        let centers = List.map (fun n -> n.center) parent.children in
        List.for_all (fun c1 ->
          List.for_all (fun c2 ->
            if c1 = c2 then true
            else
              let d = Metric.distance net.metric c1 c2 in
              d >= 0.1 *. net.alpha ** float parent.m_label *. net.root.diameter
          ) centers
        ) centers
      in
      
      diam_ok && labels_ok && separation_ok &&
      List.for_all verify_node parent.children
    in
    verify_node net.root

  (* Build complete labelled net *)
  let build net =
    let tree = build_tree_recursive net.root net in
    {net with root = tree}
end

(* Complete Packing Tree *)
module CompletePackingTree = struct
  type complete_packing_tree = {
    root: complete_packing_node;
    metric: metric_space;
    alpha: float;              (* α = 1/10 *)
    measure: measure;
    total_weight: float;       (* Total tree weight *)
  }

  (* Strict separation validation *)
  let validate_separation node1 node2 tree =
    let distance = Metric.distance tree.metric node1.center node2.center in
    let required_sep = 0.1 *. tree.alpha ** float node1.chi *. tree.root.diameter in
    
    let balls_separated =
      distance >= node1.radius +. node2.radius +. required_sep in
    
    let points_separated = 
      let v1_points = Tensor.nonzero node1.vertices |> fst in
      let v2_points = Tensor.nonzero node2.vertices |> fst in
      Tensor.iter2 (fun p1 p2 ->
        let d = Metric.distance tree.metric 
          (Tensor.get_all1 p1 |> int_of_float)
          (Tensor.get_all1 p2 |> int_of_float) in
        d >= required_sep
      ) v1_points v2_points
    in
    
    balls_separated && points_separated

  (* Greedy Separated Ball Partitioning *)
  let find_separated_partition tree node =
    let m = node.chi in
    let radius = tree.alpha ** float (m + 2) /. 4. *. tree.root.diameter in
    let excluded = ref (Tensor.zeros [tree.metric.n_points]) in
    let partitions = ref [] in
    
    while Tensor.(sum !excluded < sum node.vertices) do
      let best_center = ref (-1) in
      let best_measure = ref Float.neg_infinity in
      
      for i = 0 to tree.metric.n_points - 1 do
        if not (Tensor.get_all1 (Tensor.get1 !excluded i)) then
          let valid_center = List.for_all (fun (c, _, _) ->
            let d = Metric.distance tree.metric i c in
            d >= radius *. (1. +. tree.alpha)
          ) !partitions in
          
          if valid_center then
            let ball = Ball.create tree.metric i radius in
            let ball_points = Ball.get_points ball in
            let measure = Measure.measure_ball tree.measure ball in
            if Tensor.get_all1 measure > !best_measure then begin
              best_center := i;
              best_measure := Tensor.get_all1 measure
            end
      done;
      
      if !best_center >= 0 then begin
        let ball = Ball.create tree.metric !best_center radius in
        let vertices = Ball.get_points ball in
        excluded := Tensor.(!excluded + vertices);
        partitions := (!best_center, vertices, !best_measure) :: !partitions
      end
    done;
    
    List.rev !partitions

   (* Build complete packing tree *)
  let rec build_tree_recursive parent tree =
    let partitions = find_separated_partition tree parent in
    
    (* Create child nodes *)
    let children = List.mapi (fun i (center, vertices, measure) ->
      let diameter = Ball.diameter tree.metric vertices in
      let child = {
        vertices;
        chi = parent.chi + 2;  (* Increment by 2 for separation *)
        center;
        radius = diameter /. 2.;
        diameter;
        separation = tree.alpha ** float parent.chi *. tree.root.diameter /. 10.;
        parent_radius = parent.radius;
        children = [];
        weight = Tensor.get_all1 measure;
      } in
      
      (* Recursively build subtree *)
      if Tensor.(sum vertices == scalar 1.) then child
      else
        let subtree = build_tree_recursive child tree in
        {subtree with chi = child.chi}
    ) partitions in
    
    let total_weight = List.fold_left (fun acc c -> acc +. c.weight) 0. children in
    {parent with children; weight = total_weight}

  (* Create optimized packing tree *)
  let create metric measure =
    let alpha = 0.1 in
    let diam = Ball.diameter metric (Tensor.ones [metric.n_points]) in
    let root = {
      vertices = Tensor.ones [metric.n_points];
      chi = 0;
      center = 0;
      radius = diam /. 2.;
      diameter = diam;
      separation = 0.;
      parent_radius = 0.;
      children = [];
      weight = 1.;
    } in
    
    let tree = {root; metric; alpha; measure; total_weight = 1.} in
    let built_tree = build_tree_recursive root tree in
    {tree with root = built_tree; total_weight = built_tree.weight}

  (* Compute value of packing tree *)
  let compute_value tree h =
    let rec path_value node =
      match node.children with
      | [] -> 0.
      | cs ->
          List.fold_left (fun acc child ->
            let edge_value = 
              tree.alpha ** float node.chi *. 
              tree.root.diameter *.
              ChainingFunctional.evaluate h (1. /. float (List.length cs))
            in
            max acc (edge_value +. path_value child)
          ) 0. cs
    in
    path_value tree.root
end

(* Tree transformations and optimizations *)
module TreeTransformations = struct
  (* Tree transformation state *)
  type transform_state = {
    seen_sets: (string, bool) Hashtbl.t;  (* Track unique vertex sets *)
    memo_diameters: (string, float) Hashtbl.t;  (* Memoized diameters *)
    memo_measures: (string, float) Hashtbl.t;   (* Memoized measures *)
  }

  (* Create initial state *)
  let create_state () = {
    seen_sets = Hashtbl.create 1000;
    memo_diameters = Hashtbl.create 1000;
    memo_measures = Hashtbl.create 1000;
  }

  (* Memoized diameter computation *)
  let get_diameter state metric vertices =
    let key = Tensor.to_string vertices in
    match Hashtbl.find_opt state.memo_diameters key with
    | Some d -> d
    | None ->
        let d = Ball.diameter metric vertices in
        Hashtbl.add state.memo_diameters key d;
        d

  (* Remove redundant nodes *)
  let remove_redundant state tree =
    let rec compress node =
      match node.children with
      | [child] when Tensor.(sum (node.vertices + (~ child.vertices)) == scalar 0.) ->
          (* Single child with same vertices - merge *)
          let merged = {child with chi = min node.chi child.chi} in
          compress merged
      | cs ->
          let compressed_children = List.map compress cs in
          {node with children = compressed_children}
    in
    {tree with root = compress tree.root}

  (* Rebalance tree for better performance *)
  let rebalance state tree =
    (* Compute subtree weights *)
    let rec compute_weight node =
      match node.children with
      | [] -> 1.
      | cs ->
          let child_weights = List.map compute_weight cs in
          1. +. List.fold_left (+.) 0. child_weights
    in
    
    (* Reorder children by weight *)
    let rec rebalance_node node =
      let children = List.map (fun c -> c, compute_weight c) node.children in
      let sorted = List.sort (fun (_, w1) (_, w2) -> Float.compare w2 w1) children in
      let balanced = List.map (fun (c, _) -> rebalance_node c) (List.map fst sorted) in
      {node with children = balanced}
    in
    
    {tree with root = rebalance_node tree.root}

  (* Optimize memory usage *)
  let optimize_memory state tree =
    (* Clear unused cache entries *)
    let active_sets = Hashtbl.create 100 in
    
    let rec mark_active node =
      let key = Tensor.to_string node.vertices in
      Hashtbl.replace active_sets key true;
      List.iter mark_active node.children
    in
    mark_active tree.root;
    
    (* Remove inactive entries *)
    let cleanup tbl =
      Hashtbl.filter_map_inplace (fun k v ->
        if Hashtbl.mem active_sets k then Some v else None
      ) tbl
    in
    
    cleanup state.memo_diameters;
    cleanup state.memo_measures;
    tree
end

(* Runtime optimizations *)
module RuntimeOptimizations = struct
  (* Operation cache *)
  type operation_cache = {
    ball_measures: (int * float, float) Hashtbl.t;
    diameters: (string, float) Hashtbl.t;
    distances: (int * int, float) Hashtbl.t;
    neighbors: (int * float, int list) Hashtbl.t;
  }

  (* Create operation cache *)
  let create_cache () = {
    ball_measures = Hashtbl.create 1000;
    diameters = Hashtbl.create 1000;
    distances = Hashtbl.create 1000;
    neighbors = Hashtbl.create 1000;
  }

  (* Cached operations *)
  let cached_ball_measure cache metric measure ball =
    let key = (ball.center, ball.radius) in
    match Hashtbl.find_opt cache.ball_measures key with
    | Some m -> m
    | None ->
        let m = Measure.measure_ball measure ball |> Tensor.get_all1 in
        Hashtbl.add cache.ball_measures key m;
        m

  let cached_diameter cache metric vertices =
    let key = Tensor.to_string vertices in
    match Hashtbl.find_opt cache.diameters key with
    | Some d -> d
    | None ->
        let d = Ball.diameter metric vertices in
        Hashtbl.add cache.diameters key d;
        d

  let cached_neighbors cache metric center radius =
    let key = (center, radius) in
    match Hashtbl.find_opt cache.neighbors key with
    | Some ns -> ns
    | None ->
        let ball = Ball.create metric center radius in
        let points = Ball.get_points ball in
        let ns = Tensor.nonzero points 
                |> fst 
                |> Tensor.to_list1_int in
        Hashtbl.add cache.neighbors key ns;
        ns

  (* Batch processing *)
  let batch_process points f batch_size =
    let n = List.length points in
    let rec process acc i =
      if i >= n then List.rev acc
      else
        let batch = List.filteri (fun j _ -> j >= i && j < i + batch_size) points in
        let results = f batch in
        process (results @ acc) (i + batch_size)
    in
    process [] 0
end

(* Core integration *)
module Integration = struct
  (* Computation result *)
  type computation_result = {
    tree: complete_packing_node;
    value: float;
    valid: bool;
    error_margin: float;
  }

  (* Complete validation *)
  let validate_computation metric measure tree =
    (* Check diameter bounds *)
    let check_diameters node =
      let rec check current_node =
        let diam = Ball.diameter metric current_node.vertices in
        let valid_diam = diam <= current_node.diameter *. 1.1 in
        valid_diam && List.for_all check current_node.children
      in
      check node
    in

    (* Check separation conditions *)
    let check_separation node =
      let rec check current_node =
        let children = current_node.children in
        let pairs_valid = 
          List.for_all (fun c1 ->
            List.for_all (fun c2 ->
              if c1 == c2 then true
              else
                let d = Metric.distance metric c1.center c2.center in
                d >= c1.separation +. c2.separation
            ) children
          ) children
        in
        pairs_valid && List.for_all check children
      in
      check node
    in

    (* Check measure consistency *)
    let check_measures node =
      let rec check current_node =
        if List.length current_node.children = 0 then true
        else
          let child_sum = List.fold_left (fun acc child ->
            acc +. Measure.measure_subset measure child.vertices |> Tensor.get_all1
          ) 0. current_node.children in
          let node_measure = Measure.measure_subset measure current_node.vertices 
                           |> Tensor.get_all1 in
          abs_float (child_sum -. node_measure) <= 1e-6 &&
          List.for_all check current_node.children
      in
      check node
    in

    check_diameters tree && check_separation tree && check_measures tree

  (* Core computation *)
  let compute_core metric measure =
    let cache = RuntimeOptimizations.create_cache () in
    let sequence = CoreAlgorithms.generate_sequence metric measure in
    match CoreAlgorithms.build_tree metric measure sequence with
    | None -> None
    | Some tree ->
        let optimized = CoreAlgorithms.optimize_tree tree in
        let value = CompletePackingTree.compute_value 
          {root=optimized; metric; alpha=0.1; 
           measure; total_weight=1.} 
          (ChainingFunctional.create_gaussian()) in
        let valid = validate_computation metric measure optimized in
        Some {
          tree = optimized;
          value;
          valid;
          error_margin = 1e-6;
        }
end

(* High-level interface *)
module Interface = struct
  (* Main computation interface *)
  let compute_packing_tree distances weights =
    Utilities.validate_inputs distances weights >>= fun () ->
    Utilities.create_metric_space distances >>= fun metric ->
    Utilities.create_measure weights >>= fun measure ->
    match Integration.safe_compute metric measure with
    | Ok result -> Ok (result.tree, result.value)
    | Error msg -> Error msg

  (* Tree analysis interface *)
  let analyze_tree tree metric =
    let n = metric.n_points in
    let stats = {
      size = n;
      depth = 
        let rec max_depth node =
          match node.children with
          | [] -> 0
          | cs -> 1 + List.fold_left max 0 (List.map max_depth cs)
        in
        max_depth tree;
      branching_factor = 
        let rec avg_branching_acc node acc count =
          let child_count = List.length node.children in
          let new_acc = acc +. float child_count in
          let new_count = count + 1 in
          List.fold_left (fun (a, c) child ->
            let (ca, cc) = avg_branching_acc child a c in
            (ca, cc)
          ) (new_acc, new_count) node.children
        in
        let (total, count) = avg_branching_acc tree 0. 0 in
        if count > 0 then total /. float count else 0.;
      diameter = tree.diameter;
      separation = tree.separation;
    } in
    Ok stats

  (* Convenience functions *)
  let create_gaussian_tree distances weights =
    compute_packing_tree distances weights >>= fun (tree, _) ->
    let h = ChainingFunctional.create_gaussian () in
    Utilities.compute_tree_value tree h

  let optimize_existing_tree tree metric measure =
    let state = TreeTransformations.create_state () in
    let optimized = TreeTransformations.remove_redundant state tree in
    let balanced = TreeTransformations.rebalance state optimized in
    TreeTransformations.optimize_memory state balanced
end

(* Final validation and analysis *)
module Analysis = struct
  (* Tree statistics *)
  type tree_stats = {
    size: int;
    depth: int;
    branching_factor: float;
    diameter: float;
    separation: float;
  }

  (* Compute tree properties *)
  let compute_properties tree =
    let rec compute node acc =
      let curr_stats = {
        node_count = 1;
        leaf_count = if node.children = [] then 1 else 0;
        total_depth = node.chi;
        max_depth = node.chi;
        total_separation = node.separation;
      } in
      List.fold_left (fun stats child ->
        let child_stats = compute child stats in
        {
          node_count = stats.node_count + child_stats.node_count;
          leaf_count = stats.leaf_count + child_stats.leaf_count;
          total_depth = stats.total_depth + child_stats.total_depth;
          max_depth = max stats.max_depth child_stats.max_depth;
          total_separation = stats.total_separation +. child_stats.total_separation;
        }
      ) curr_stats node.children
    in
    let stats = compute tree {
      node_count = 0;
      leaf_count = 0;
      total_depth = 0;
      max_depth = 0;
      total_separation = 0.;
    } in
    {
      size = stats.node_count;
      depth = stats.max_depth;
      branching_factor = float stats.node_count /. float (max 1 (stats.node_count - stats.leaf_count));
      diameter = tree.diameter;
      separation = stats.total_separation /. float stats.node_count;
    }

  (* Performance analysis *)
  let analyze_performance f input_size =
    let start_time = Unix.gettimeofday () in
    let result = f () in
    let end_time = Unix.gettimeofday () in
    let time_taken = end_time -. start_time in
    {
      input_size;
      time_taken;
      result = result;
    }
end