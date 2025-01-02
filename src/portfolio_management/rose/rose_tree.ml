let create_leaf indices region stats =
  {
    data_indices = indices;
    region = region;
    split_var = None;
    split_point = None;
    left = None;
    right = None;
    stats = stats;
  }

let compute_node_weights node z =
  let rec traverse node weight =
    match (node.split_var, node.split_point) with
    | (None, None) -> weight /. float_of_int (Array.length node.data_indices)
    | (Some var, Some split) ->
        if z.(var) <= split then
          traverse (Option.get node.left) (weight /. 2.0)
        else
          traverse (Option.get node.right) (weight /. 2.0)
    | _ -> failwith "Invalid tree state"
  in
  traverse node 1.0

let build_tree data nuisance theta indices max_depth mtry min_node_size =
  let rec build node depth =
    if depth >= max_depth || Array.length node.data_indices < min_node_size then
      node
    else
      let (split_var, split_point) = Utils.find_best_split 
        data nuisance theta node mtry in
      
      let (left_indices, right_indices) = 
        Utils.partition_data node.data_indices split_var split_point data in
      
      if Array.length left_indices = 0 || Array.length right_indices = 0 then
        node
      else
        let left_region = Array.copy node.region in
        let right_region = Array.copy node.region in
        left_region.(split_var) <- (fst node.region.(split_var), split_point);
        right_region.(split_var) <- (split_point, snd node.region.(right_region.(split_var) <- (split_point, snd node.region.(split_var));
        
        let left_stats = {
          d_theta_sum = Array.fold_left (fun acc i ->
            let (_, d_theta) = Models.compute_influence 
              Models.PartiallyLinear data.(i) nuisance.(i) theta in
            acc +. d_theta
          ) 0.0 left_indices;
          psi_sq_sum = Array.fold_left (fun acc i ->
            let (psi, _) = Models.compute_influence 
              Models.PartiallyLinear data.(i) nuisance.(i) theta in
            acc +. psi *. psi
          ) 0.0 left_indices;
          n = Array.length left_indices;
        } in
        
        let right_stats = {
          d_theta_sum = Array.fold_left (fun acc i ->
            let (_, d_theta) = Models.compute_influence 
              Models.PartiallyLinear data.(i) nuisance.(i) theta in
            acc +. d_theta
          ) 0.0 right_indices;
          psi_sq_sum = Array.fold_left (fun acc i ->
            let (psi, _) = Models.compute_influence 
              Models.PartiallyLinear data.(i) nuisance.(i) theta in
            acc +. psi *. psi
          ) 0.0 right_indices;
          n = Array.length right_indices;
        } in
        
        let left = create_leaf left_indices left_region left_stats in
        let right = create_leaf right_indices right_region right_stats in
        
        let left_node = build left (depth + 1) in
        let right_node = build right (depth + 1) in
        
        {
          data_indices = node.data_indices;
          region = node.region;
          split_var = Some split_var;
          split_point = Some split_point;
          left = Some left_node;
          right = Some right_node;
          stats = node.stats;
        }
  in
  
  let init_region = Array.make (Array.length data.(0).z) (neg_infinity, infinity) in
  let init_stats = {
    d_theta_sum = Array.fold_left (fun acc i ->
      let (_, d_theta) = Models.compute_influence 
        Models.PartiallyLinear data.(i) nuisance.(i) theta in
      acc +. d_theta
    ) 0.0 indices;
    psi_sq_sum = Array.fold_left (fun acc i ->
      let (psi, _) = Models.compute_influence 
        Models.PartiallyLinear data.(i) nuisance.(i) theta in
      acc +. psi *. psi
    ) 0.0 indices;
    n = Array.length indices;
  } in
  
  let root = create_leaf indices init_region init_stats in
  build root 0