module Node = struct
  type t = int
  let compare = Int.compare
end

module NodeSet = Set.Make(Node)
module NodeMap = Map.Make(Node)

type t = {
  nodes: NodeSet.t;
  parents: NodeSet.t NodeMap.t;
  max_depth: int;
  max_in_degree: int;
  effective_max_in_degree: int;
  effective_max_depth: int;
}

let get_parents g node =
  NodeMap.find_opt node g.parents 
  |> Option.value ~default:NodeSet.empty

let validate_topological_order parents =
  let dim = NodeMap.cardinal parents in
  let orders = Array.make dim (-1) in
  let visited = Array.make dim false in
  
  let rec visit node order =
    if visited.(node) then
      orders.(node) >= 0
    else begin
      visited.(node) <- true;
      let pa = NodeMap.find node parents in
      let valid = NodeSet.for_all (fun p ->
        visit p (order - 1)
      ) pa in
      if valid then orders.(node) <- order;
      valid
    end
  in
  
  let rec try_order node =
    if node = dim then true
    else if visit node node then try_order (node + 1)
    else false
  in
  try_order 0

let compute_effective_bounds g reward_node =
  let rec get_ancestors_with_depth node depth acc =
    let pa = NodeMap.find node g.parents in
    NodeSet.fold (fun p acc ->
      let acc = NodeMap.add p depth acc in
      get_ancestors_with_depth p (depth + 1) acc
    ) pa acc
  in
  
  let ancestor_depths = get_ancestors_with_depth reward_node 1 NodeMap.empty in
  
  let effective_max_depth =
    NodeMap.fold (fun _ depth max_d ->
      max max_d depth
    ) ancestor_depths 0
  in
  
  let effective_max_in_degree =
    NodeMap.fold (fun node _ max_d ->
      let d = NodeSet.cardinal (NodeMap.find node g.parents) in
      max max_d d
    ) ancestor_depths 0
  in
  
  (effective_max_depth, effective_max_in_degree)

let create nodes parents =
  let max_in_degree = 
    NodeMap.fold (fun _ parents max_d -> 
      max max_d (NodeSet.cardinal parents)
    ) parents 0
  in
  let reward_node = NodeSet.max_elt nodes in
  let effective_max_depth, effective_max_in_degree = 
    compute_effective_bounds 
      {nodes; parents; max_depth=0; max_in_degree; 
       effective_max_in_degree=0; effective_max_depth=0} 
      reward_node
  in
  {
    nodes;
    parents;
    max_depth = effective_max_depth;  (* Using effective as max *)
    max_in_degree;
    effective_max_in_degree;
    effective_max_depth;
  }