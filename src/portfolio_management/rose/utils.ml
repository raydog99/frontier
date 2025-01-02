open Types

let compute_split_criterion data nuisance theta indices feature split_point =
  let (left_indices, right_indices) = partition_data indices feature split_point data in
  
  let compute_stats indices =
    let stats = ref {d_theta_sum = 0.0; psi_sq_sum = 0.0; n = 0} in
    Array.iter (fun i ->
      let (psi, d_theta) = Models.compute_influence 
        Models.PartiallyLinear data.(i) nuisance.(i) theta in
      stats := {
        d_theta_sum = !stats.d_theta_sum +. d_theta;
        psi_sq_sum = !stats.psi_sq_sum +. psi *. psi;
        n = !stats.n + 1
      }
    ) indices;
    !stats
  in
  
  let left_stats = compute_stats left_indices in
  let right_stats = compute_stats right_indices in
  let parent_stats = compute_stats indices in
  
  (left_stats.d_theta_sum ** 2.0) /. left_stats.psi_sq_sum +.
  (right_stats.d_theta_sum ** 2.0) /. right_stats.psi_sq_sum -.
  (parent_stats.d_theta_sum ** 2.0) /. parent_stats.psi_sq_sum

let find_best_split data nuisance theta node mtry =
  let n_features = Array.length data.(0).z in
  let feature_indices = Array.init n_features (fun i -> i) in
  Array.shuffle feature_indices;
  
  let best_score = ref neg_infinity in
  let best_var = ref 0 in
  let best_point = ref 0.0 in
  
  for i = 0 to min mtry n_features - 1 do
    let feat_idx = feature_indices.(i) in
    let sorted_values = 
      Array.map (fun i -> data.(i).z.(feat_idx)) node.data_indices
      |> Array.to_list 
      |> List.sort_uniq compare in
    
    List.iter (fun split_point ->
      let score = compute_split_criterion data nuisance theta 
                   node.data_indices feat_idx split_point in
      if score > !best_score then (
        best_score := score;
        best_var := feat_idx;
        best_point := split_point
      )
    ) sorted_values
  done;
  
  (!best_var, !best_point)

let partition_data indices split_var split_point data =
  Array.partition (fun i -> data.(i).z.(split_var) <= split_point) indices