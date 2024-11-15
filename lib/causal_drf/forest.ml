open Torch

type split_stats = {
  left_treatment_prop: float;
  right_treatment_prop: float;
  left_size: int;
  right_size: int;
  left_kernel_matrix: Tensor.t;
  right_kernel_matrix: Tensor.t;
}

type tree_params = {
  max_depth: int;
  min_samples_leaf: int;
  max_features: int;
  honesty_fraction: float;
  regularization: float;
}

let random_feature_subset n_features max_features =
  let features = Array.init n_features (fun i -> i) in
  let subset_size = min max_features n_features in
  
  for i = 0 to subset_size - 1 do
    let j = i + Random.int (n_features - i) in
    let temp = features.(i) in
    features.(i) <- features.(j);
    features.(j) <- temp
  done;
  Array.sub features 0 subset_size

let calculate_node_stats data indices =
  let n = Array.length indices in
  let treatment = Array.map (fun i -> 
    Tensor.get_float1 data.treatment i) indices in
  let outcomes = Array.map (fun i -> 
    Tensor.get data.outcome i) indices in
  
  let treatment_mean = Array.fold_left (+.) 0. treatment /. float_of_int n in
  let outcome_mean = Array.fold_left (fun acc o -> 
    Tensor.add acc o) (Tensor.zeros_like (Array.get outcomes 0)) outcomes in
  let outcome_mean = Tensor.div outcome_mean (float_of_int n) in
  
  let k_mat = Tensor.zeros [|n; n|] in
  for i = 0 to n-1 do
    for j = i to n-1 do
      let k_val = evaluate_kernel (Gaussian 1.0) outcomes.(i) outcomes.(j) in
      Tensor.set k_mat [|i; j|] k_val;
      if i <> j then Tensor.set k_mat [|j; i|] k_val
    done
  done;
  
  {
    left_treatment_prop = treatment_mean;
    right_treatment_prop = 1.0 -. treatment_mean;
    left_size = Array.length indices;
    right_size = 0;
    left_kernel_matrix = k_mat;
    right_kernel_matrix = Tensor.zeros [|1; 1|];
  }

let evaluate_split data left_indices right_indices =
  let left_stats = calculate_node_stats data left_indices in
  let right_stats = calculate_node_stats data right_indices in
  
  let n_left = Array.length left_indices in
  let n_right = Array.length right_indices in
  let total_n = n_left + n_right in
  
  (* Calculate MMD statistic *)
  let calc_mmd stats indices =
    let weights = Array.map (fun i ->
      if Tensor.get_float1 data.treatment i > 0.5 then
        1.0 /. stats.left_treatment_prop
      else
        -1.0 /. stats.right_treatment_prop
    ) indices in
    
    let weighted_sum = ref 0.0 in
    let n = Array.length indices in
    for i = 0 to n-1 do
      for j = 0 to n-1 do
        weighted_sum := !weighted_sum +. 
          weights.(i) *. weights.(j) *. 
          Tensor.get_float2 stats.left_kernel_matrix i j
      done
    done;
    !weighted_sum
  in
  
  let left_mmd = calc_mmd left_stats left_indices in
  let right_mmd = calc_mmd right_stats right_indices in
  
  let weighted_mmd = 
    (float_of_int n_left *. left_mmd +. 
     float_of_int n_right *. right_mmd) /.
    float_of_int total_n in
  
  let balance = 
    min left_stats.left_treatment_prop (1.0 -. left_stats.left_treatment_prop) +.
    min right_stats.left_treatment_prop (1.0 -. right_stats.left_treatment_prop) in
  
  weighted_mmd *. 0.7 +. balance *. 0.3

let find_best_split data indices params =
  let n = Array.length indices in
  let feature_subset = random_feature_subset 
    (Tensor.size2 data.features 1) params.max_features in
  
  let best_split = ref None in
  let best_score = ref neg_infinity in
  
  Array.iter (fun feature ->
    (* Get sorted feature values *)
    let sorted_values = 
      Array.map (fun i -> 
        (i, Tensor.get_float2 data.features i feature)) indices
      |> Array.sort (fun (_, a) (_, b) -> compare a b) in
    
    (* Try potential split points *)
    for i = params.min_samples_leaf to n - params.min_samples_leaf - 1 do
      let split_val = 
        (snd sorted_values.(i) +. snd sorted_values.(i+1)) /. 2.0 in
      
      let left_indices, right_indices = Array.partition (fun i ->
        Tensor.get_float2 data.features i feature < split_val
      ) indices in
      
      if Array.length left_indices >= params.min_samples_leaf && 
         Array.length right_indices >= params.min_samples_leaf then begin
        let score = evaluate_split data left_indices right_indices in
        if score > !best_score then begin
          best_score := score;
          best_split := Some (feature, split_val, left_indices, right_indices)
        end
      end
    done
  ) feature_subset;
  !best_split

let build_tree data params =
  let rec split_node indices depth =
    if depth >= params.max_depth || 
       Array.length indices <= params.min_samples_leaf then
      {
        split_feature = None;
        split_value = None;
        left_child = None;
        right_child = None;
        samples = indices;
      }
    else
      match find_best_split data indices params with
      | None -> 
          {
            split_feature = None;
            split_value = None;
            left_child = None;
            right_child = None;
            samples = indices;
          }
      | Some (feature, value, left_indices, right_indices) ->
          {
            split_feature = Some feature;
            split_value = Some value;
            left_child = Some (split_node left_indices (depth + 1));
            right_child = Some (split_node right_indices (depth + 1));
            samples = indices;
          }
  in
  
  let root_indices = Array.init (Tensor.size2 data.features 0) (fun i -> i) in
  split_node root_indices 0

let build_forest data n_trees n_groups =
  let params = {
    max_depth = 10;
    min_samples_leaf = 5;
    max_features = Tensor.size2 data.features 1 / 3;
    honesty_fraction = 0.5;
    regularization = 1.0;
  } in
  
  let trees = Array.init n_trees (fun _ ->
    build_tree data params
  ) in
  
  {trees; n_trees; n_groups}

let get_leaf_node tree x =
  let rec traverse node =
    match node.split_feature, node.split_value with
    | None, None -> node
    | Some feature, Some value ->
        if Tensor.get_float1 x feature < value then
          match node.left_child with
          | Some left -> traverse left
          | None -> node
        else
          match node.right_child with 
          | Some right -> traverse right
          | None -> node
    | _ -> node
  in traverse tree

let get_weights forest x n =
  let weights = Array.make n 0.0 in
  Array.iter (fun tree ->
    let leaf = get_leaf_node tree x in
    let leaf_size = float_of_int (Array.length leaf.samples) in
    Array.iter (fun i -> 
      weights.(i) <- weights.(i) +. 1.0 /. (leaf_size *. float_of_int forest.n_trees)
    ) leaf.samples
  ) forest.trees;
  weights

let predict forest x =
  let n = match forest.trees with
    | [||] -> 0
    | trees -> Array.length trees.(0).samples
  in
  let weights = get_weights forest x n in
  let prediction = ref (Tensor.zeros [|1|]) in
  for i = 0 to n-1 do
    prediction := Tensor.add !prediction 
      (Tensor.mul_scalar (Tensor.get forest.trees.(0).samples i) weights.(i))
  done;
  !prediction

let validate_prediction forest x =
  let n_features = Tensor.size2 x 1 in
  let expected_features = match forest.trees with
    | [||] -> 0
    | trees -> Tensor.size2 trees.(0).samples 1
  in
  if n_features <> expected_features then
    Error "Mismatched feature dimensions"
  else
    Ok ()

let calculate_node_variance data indices =
  let n = Array.length indices in
  if n = 0 then 0.0 else
    let mean = Array.fold_left (fun acc i ->
      Tensor.add acc (Tensor.get data.outcome i)
    ) (Tensor.zeros_like (Tensor.get data.outcome 0)) indices
    |> fun t -> Tensor.div t (float_of_int n) in
    
    let var_sum = Array.fold_left (fun acc i ->
      let diff = Tensor.sub (Tensor.get data.outcome i) mean in
      acc +. (Tensor.float_value (Tensor.dot diff diff))
    ) 0.0 indices in
    
    var_sum /. float_of_int n