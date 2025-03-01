open Torch

(* Helper for computing variance of a tensor *)
let variance tensor =
    let mean = Tensor.mean tensor in
    let n = Tensor.shape tensor |> List.hd |> float_of_int in
    let sum_squared_diff = Tensor.( ((tensor - mean) ** (Scalar.f 2.)) |> sum ) in
    Tensor.(sum_squared_diff / (Scalar.f (n -. 1.)))

(* Compute conditional expectation E[Y | X \in A] *)
let conditional_expectation y condition =
    let filtered_y = Tensor.masked_select y condition in
    if Tensor.shape filtered_y |> List.hd > 0 then
      Tensor.mean filtered_y
    else
      Tensor.zeros [1]

(* Compute conditional variance Var(Y | X \in A) *)
let conditional_variance y condition =
    let filtered_y = Tensor.masked_select y condition in
    if Tensor.shape filtered_y |> List.hd > 0 then
      variance filtered_y
    else
      Tensor.zeros [1]
      
(* L1 norm - mean absolute deviation *)
let l1_deviation tensor =
    let mean = Tensor.mean tensor in
    Tensor.((tensor - mean) |> abs |> mean)
    
(* Create a mask for 1D tensor where x_j < threshold *)
let mask_lt x j threshold =
    let x_j = Tensor.select x ~dim:1 ~index:j in
    Tensor.(x_j < (Scalar.f threshold))
    
(* Create a mask for 1D tensor where x_j >= threshold *)
let mask_ge x j threshold =
    let x_j = Tensor.select x ~dim:1 ~index:j in
    Tensor.(x_j >= (Scalar.f threshold))

(* Partition-based martingale approximations *)
module Martingale = struct
  type split_rule = 
    | Variance
    | Simons 
    | Minimax
    | Median
    
  (* Find the best split point using the Variance rule *)
  let find_variance_split u range =
    let a, b = range in
    let n = 100 in  (* Number of candidate split points *)
    let candidates = Array.init n (fun i -> a +. (b -. a) *. (float_of_int i) /. (float_of_int n)) in
    
    let u_tensor = Tensor.of_float1 u in
    
    let losses = Array.map (fun threshold ->
      let lt_mask = Tensor.(u_tensor < (Scalar.f threshold)) in
      let ge_mask = Tensor.(u_tensor >= (Scalar.f threshold)) in
      
      let p_lt = Tensor.sum lt_mask |> Tensor.item |> Float.to_float 
                 /. (Array.length u |> float_of_int) in
      let p_ge = Tensor.sum ge_mask |> Tensor.item |> Float.to_float
                 /. (Array.length u |> float_of_int) in
                 
      let var_lt = conditional_variance u_tensor lt_mask |> Tensor.item |> Float.to_float in
      let var_ge = conditional_variance u_tensor ge_mask |> Tensor.item |> Float.to_float in
      
      p_lt *. var_lt +. p_ge *. var_ge
    ) candidates in
    
    let min_idx = ref 0 in
    let min_val = ref losses.(0) in
    for i = 1 to n - 1 do
      if losses.(i) < !min_val then begin
        min_val := losses.(i);
        min_idx := i
      end
    done;
    
    candidates.(!min_idx)
    
  (* Find the best split point using the Simons rule *)
  let find_simons_split u range =
    let u_mean = Array.fold_left (+.) 0. u /. (float_of_int (Array.length u)) in
    u_mean
    
  (* Find the best split point using the Minimax rule *)
  let find_minimax_split u range =
    let a, b = range in
    let n = 100 in  (* Number of candidate split points *)
    let candidates = Array.init n (fun i -> a +. (b -. a) *. (float_of_int i) /. (float_of_int n)) in
    
    let u_tensor = Tensor.of_float1 u in
    
    let max_losses = Array.map (fun threshold ->
      let lt_mask = Tensor.(u_tensor < (Scalar.f threshold)) in
      let ge_mask = Tensor.(u_tensor >= (Scalar.f threshold)) in
      
      let p_lt = Tensor.sum lt_mask |> Tensor.item |> Float.to_float 
                 /. (Array.length u |> float_of_int) in
      let p_ge = Tensor.sum ge_mask |> Tensor.item |> Float.to_float
                 /. (Array.length u |> float_of_int) in
                 
      let var_lt = conditional_variance u_tensor lt_mask |> Tensor.item |> Float.to_float in
      let var_ge = conditional_variance u_tensor ge_mask |> Tensor.item |> Float.to_float in
      
      max p_lt *. var_lt p_ge *. var_ge
    ) candidates in
    
    let min_idx = ref 0 in
    let min_val = ref max_losses.(0) in
    for i = 1 to n - 1 do
      if max_losses.(i) < !min_val then begin
        min_val := max_losses.(i);
        min_idx := i
      end
    done;
    
    candidates.(!min_idx)
    
  (* Find the best split point using the Median rule *)
  let find_median_split u range =
    let sorted = Array.copy u in
    Array.sort compare sorted;
    let n = Array.length sorted in
    if n mod 2 = 0 then
      (sorted.(n/2) +. sorted.(n/2 - 1)) /. 2.0
    else
      sorted.(n/2)
    
  (* Create Mk for a given k - the martingale approximation *)
  let create_martingale u rule k =
    let rec build_partitions partitions level =
      if level >= k then partitions
      else
        let new_partitions = List.flatten (
          List.map (fun (a, b) ->
            let split_point = match rule with
              | Variance -> find_variance_split u (a, b)
              | Simons -> find_simons_split u (a, b)
              | Minimax -> find_minimax_split u (a, b)
              | Median -> find_median_split u (a, b)
            in
            [(a, split_point); (split_point, b)]
          ) partitions
        ) in
        build_partitions new_partitions (level + 1)
    in
    
    let partitions = build_partitions [(-1e10, 1e10)] 0 in
    
    let partition_means = List.map (fun (a, b) ->
      let partition_u = Array.of_list (
        List.filter_map (fun (value, idx) ->
          if value >= a && value < b then Some value else None
        ) (List.combine (Array.to_list u) (List.init (Array.length u) (fun i -> i)))
      ) in
      
      let mean = 
        if Array.length partition_u > 0 then
          Array.fold_left (+.) 0.0 partition_u /. float_of_int (Array.length partition_u)
        else
          (a +. b) /. 2.0
      in
      
      ((a, b), mean)
    ) partitions in
    
    (* Return a function that maps a value to its conditional expectation *)
    fun value ->
      try
        let ((_, _), mean) = List.find (fun ((a, b), _) -> value >= a && value < b) partition_means in
        mean
      with Not_found -> 
        (* If no partition found, return the overall mean *)
        Array.fold_left (+.) 0.0 u /. float_of_int (Array.length u)
end
    
(* Generate a sinusoidal dataset *)
let generate_sinusoidal_data n noise_var =
    let x = Tensor.linspace ~start:(Scalar.f 0.) ~end_:(Scalar.f 1.) ~steps:n in
    let true_fn = Tensor.(sin (x * (Scalar.f (4. *. Float.pi)))) in
    let noise = Tensor.randn [n] ~mean:(Scalar.f 0.) ~std:(Scalar.f (sqrt noise_var)) in
    let y = Tensor.(true_fn + noise) in
    
    (* Reshape x to be [n, 1] for decision tree input *)
    let x_reshaped = Tensor.reshape x ~shape:[n; 1] in
    
    x_reshaped, y, true_fn
    
(* Generate a 2D function dataset *)
let generate_2d_function_data n =
    let x1 = Tensor.rand [n] in
    let x2 = Tensor.rand [n] in
    
    (* Compute the function values *)
    let u1 = Tensor.((x1 - (Scalar.f 0.5)) * (Scalar.f 3.0)) in
    let u2 = Tensor.((x2 - (Scalar.f 0.5)) * (Scalar.f 3.0)) in
    
    let term1 = Tensor.((Scalar.f 2.0) - ((Scalar.f 2.1) * (u1 ** (Scalar.f 2.0))) + (u1 ** (Scalar.f 4.0)) / (Scalar.f 3.0)) in
    let term2 = Tensor.(u1 * u2) in
    let term3 = Tensor.(((Scalar.f (-4.0)) + ((Scalar.f 4.0) * (u2 ** (Scalar.f 2.0)))) * (u2 ** (Scalar.f 2.0))) in
    
    let y = Tensor.(term1 * (u1 ** (Scalar.f 2.0)) + term2 + term3) in
    
    (* Combine x1 and x2 into a single feature tensor *)
    let x = Tensor.stack [x1; x2] ~dim:1 in
    
    x, y dataset

let generate_2d_function_data n =
    let x1 = Tensor.rand [n] in
    let x2 = Tensor.rand [n] in
    
    (* Compute the function values *)
    let u1 = Tensor.((x1 - (Scalar.f 0.5)) * (Scalar.f 3.0)) in
    let u2 = Tensor.((x2 - (Scalar.f 0.5)) * (Scalar.f 3.0)) in
    
    let term1 = Tensor.((Scalar.f 2.0) - ((Scalar.f 2.1) * (u1 ** (Scalar.f 2.0))) + (u1 ** (Scalar.f 4.0)) / (Scalar.f 3.0)) in
    let term2 = Tensor.(u1 * u2) in
    let term3 = Tensor.(((Scalar.f (-4.0)) + ((Scalar.f 4.0) * (u2 ** (Scalar.f 2.0)))) * (u2 ** (Scalar.f 2.0))) in
    
    let y = Tensor.(term1 * (u1 ** (Scalar.f 2.0)) + term2 + term3) in
    
    (* Combine x1 and x2 into a single feature tensor *)
    let x = Tensor.stack [x1; x2] ~dim:1 in
    
    x, y
  
(* Split data into training and test sets *)
let train_test_split x y test_ratio =
    let n = Tensor.shape x |> List.hd in
    let n_test = int_of_float (float_of_int n *. test_ratio) in
    let n_train = n - n_test in
    
    (* Create random permutation indices *)
    let indices = Array.init n (fun i -> i) in
    for i = n - 1 downto 1 do
      let j = Random.int (i + 1) in
      let temp = indices.(i) in
      indices.(i) <- indices.(j);
      indices.(j) <- temp;
    done;
    
    (* Split indices *)
    let train_indices = Array.sub indices 0 n_train in
    let test_indices = Array.sub indices n_train n_test in
    
    (* Create training data *)
    let x_train = Tensor.stack (
      Array.map (fun i -> Tensor.select x ~dim:0 ~index:i) train_indices
      |> Array.to_list
    ) ~dim:0 in
    
    let y_train = Tensor.stack (
      Array.map (fun i -> Tensor.select y ~dim:0 ~index:i) train_indices
      |> Array.to_list
    ) ~dim:0 in
    
    (* Create test data *)
    let x_test = Tensor.stack (
      Array.map (fun i -> Tensor.select x ~dim:0 ~index:i) test_indices
      |> Array.to_list
    ) ~dim:0 in
    
    let y_test = Tensor.stack (
      Array.map (fun i -> Tensor.select y ~dim:0 ~index:i) test_indices
      |> Array.to_list
    ) ~dim:0 in
    
    (x_train, y_train), (x_test, y_test)


(* Decision tree algorithms *)
module DecisionTree = struct
  type split_criterion = 
    | VarianceSplit
    | MinimaxSplit
    | CyclicMinimaxSplit
    | L1VarianceSplit
    | L1MinimaxSplit
    | L1CyclicMinimaxSplit
    
  type node = {
    region: (float * float) array;  (* min and max for each dimension *)
    value: float;                   (* prediction value for this node *)
    feature: int option;            (* split feature index *)
    threshold: float option;        (* split threshold *)
    left: node option;              (* left child *)
    right: node option;             (* right child *)
    samples: int;                   (* number of samples in this node *)
    depth: int;                     (* depth of this node in the tree *)
    mse: float;                     (* mean squared error at this node *)
  }
  
  (* Metrics for node quality *)
  type node_metrics = {
    n_samples: int;             (* number of samples *)
    n_leaves: int;              (* number of leaf nodes *)
    max_depth: int;             (* maximum depth *)
    avg_depth: float;           (* average depth of leaves *)
    mse: float;                 (* mean squared error *)
    tv_norm: float option;      (* total variation norm estimate *)
  }
  
  (* Create a new node *)
  let create_node region value ~samples ~depth ~mse = {
    region;
    value;
    feature = None;
    threshold = None;
    left = None;
    right = None;
    samples;
    depth;
    mse;
  }
  
  (* Calculate node metrics for the entire tree *)
  let calc_node_metrics tree =
    let n_samples = tree.samples in
    
    let rec traverse node acc =
      let (n_leaves, sum_depth, max_depth, mse_weighted_sum) = acc in
      
      match node.feature with
      | None -> 
          (* Leaf node *)
          (n_leaves + 1, 
           sum_depth + node.depth, 
           max max_depth node.depth,
           mse_weighted_sum +. (float_of_int node.samples *. node.mse))
      | Some _ ->
          (* Internal node - traverse children *)
          let acc' = match node.left with
            | Some left -> traverse left acc
            | None -> acc
          in
          
          match node.right with
            | Some right -> traverse right acc'
            | None -> acc'
    in
    
    let (n_leaves, sum_depth, max_depth, mse_weighted_sum) = 
      traverse tree (0, 0, 0, 0.0) 
    in
    
    let avg_depth = float_of_int sum_depth /. float_of_int n_leaves in
    let mse = mse_weighted_sum /. float_of_int n_samples in
    
    (* Try to estimate TV norm based on the tree structure *)
    let tv_norm_estimate = 
      if max_depth <= 0 then None
      else
        (* Rough estimate based on number of leaf nodes and range of y *)
        let n_splits = n_leaves - 1 in
        Some (float_of_int n_splits *. sqrt(mse) *. 2.0)
    in
    
    {
      n_samples;
      n_leaves;
      max_depth;
      avg_depth;
      mse;
      tv_norm = tv_norm_estimate;
    }
  
  (* Check if a set is splittable *)
  let is_splittable x y region =
    let n = Tensor.shape x |> List.hd in
    
    (* Check if there are any samples in the region *)
    if n = 0 then false
    else
      (* Check if Y is constant *)
      let y_variance = variance y in
      if Tensor.item y_variance |> Float.to_float = 0.0 then false
      else
        (* Check if X is constant in every dimension *)
        let d = Tensor.shape x |> List.tl |> List.hd in
        let x_constant = ref true in
        for j = 0 to d - 1 do
          let x_j = Tensor.select x ~dim:1 ~index:j in
          let x_j_min = Tensor.min x_j |> fst |> Tensor.item |> Float.to_float in
          let x_j_max = Tensor.max x_j |> fst |> Tensor.item |> Float.to_float in
          if Float.abs (x_j_max -. x_j_min) > 1e-6 then
            x_constant := false
        done;
        not !x_constant
        
  (* Find the best split according to VarianceSplit criterion *)
  let find_variance_split x y =
    let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
    
    let best_feature = ref 0 in
    let best_threshold = ref 0.0 in
    let best_score = ref Float.infinity in
    
    for j = 0 to d - 1 do
      let x_j = Tensor.select x ~dim:1 ~index:j in
      let min_val = Tensor.min x_j |> fst |> Tensor.item |> Float.to_float in
      let max_val = Tensor.max x_j |> fst |> Tensor.item |> Float.to_float in
      
      (* Try different thresholds *)
      let step = (max_val -. min_val) /. 10.0 in
      for t = 1 to 9 do
        let threshold = min_val +. step *. float_of_int t in
        
        let lt_mask = mask_lt x j threshold in
        let ge_mask = mask_ge x j threshold in
        
        let n_lt = Tensor.sum lt_mask |> Tensor.item |> Float.to_float in
        let n_ge = Tensor.sum ge_mask |> Tensor.item |> Float.to_float in
        
        if n_lt > 0.0 && n_ge > 0.0 then begin
          let var_lt = conditional_variance y lt_mask |> Tensor.item |> Float.to_float in
          let var_ge = conditional_variance y ge_mask |> Tensor.item |> Float.to_float in
          
          let p_lt = n_lt /. float_of_int n in
          let p_ge = n_ge /. float_of_int n in
          
          let score = p_lt *. var_lt +. p_ge *. var_ge in
          
          if score < !best_score then begin
            best_score := score;
            best_feature := j;
            best_threshold := threshold
          end
        end
      done
    done;
    
    !best_feature, !best_threshold, !best_score
    
  (* Find the best split according to MinimaxSplit criterion *)
  let find_minimax_split x y =
    let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
    
    let best_feature = ref 0 in
    let best_threshold = ref 0.0 in
    let best_score = ref Float.infinity in
    
    for j = 0 to d - 1 do
      let x_j = Tensor.select x ~dim:1 ~index:j in
      let min_val = Tensor.min x_j |> fst |> Tensor.item |> Float.to_float in
      let max_val = Tensor.max x_j |> fst |> Tensor.item |> Float.to_float in
      
      (* Try different thresholds *)
      let step = (max_val -. min_val) /. 10.0 in
      for t = 1 to 9 do
        let threshold = min_val +. step *. float_of_int t in
        
        let lt_mask = mask_lt x j threshold in
        let ge_mask = mask_ge x j threshold in
        
        let n_lt = Tensor.sum lt_mask |> Tensor.item |> Float.to_float in
        let n_ge = Tensor.sum ge_mask |> Tensor.item |> Float.to_float in
        
        if n_lt > 0.0 && n_ge > 0.0 then begin
          let var_lt = conditional_variance y lt_mask |> Tensor.item |> Float.to_float in
          let var_ge = conditional_variance y ge_mask |> Tensor.item |> Float.to_float in
          
          let p_lt = n_lt /. float_of_int n in
          let p_ge = n_ge /. float_of_int n in
          
          (* Maximize between the two variances *)
          let score = max (p_lt *. var_lt) (p_ge *. var_ge) in
          
          if score < !best_score then begin
            best_score := score;
            best_feature := j;
            best_threshold := threshold
          end
        end
      done
    done;
    
    !best_feature, !best_threshold, !best_score
    
  (* Find the best split according to CyclicMinimaxSplit criterion *)
  let find_cyclic_minimax_split x y j =
    let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
    
    let best_threshold = ref 0.0 in
    let best_score = ref Float.infinity in
    
    let x_j = Tensor.select x ~dim:1 ~index:j in
    let min_val = Tensor.min x_j |> fst |> Tensor.item |> Float.to_float in
    let max_val = Tensor.max x_j |> fst |> Tensor.item |> Float.to_float in
    
    (* Try different thresholds *)
    let step = (max_val -. min_val) /. 10.0 in
    for t = 1 to 9 do
      let threshold = min_val +. step *. float_of_int t in
      
      let lt_mask = mask_lt x j threshold in
      let ge_mask = mask_ge x j threshold in
      
      let n_lt = Tensor.sum lt_mask |> Tensor.item |> Float.to_float in
      let n_ge = Tensor.sum ge_mask |> Tensor.item |> Float.to_float in
      
      if n_lt > 0.0 && n_ge > 0.0 then begin
        let var_lt = conditional_variance y lt_mask |> Tensor.item |> Float.to_float in
        let var_ge = conditional_variance y ge_mask |> Tensor.item |> Float.to_float in
        
        let p_lt = n_lt /. float_of_int n in
        let p_ge = n_ge /. float_of_int n in
        
        (* Maximize between the two variances *)
        let score = max (p_lt *. var_lt) (p_ge *. var_ge) in
        
        if score < !best_score then begin
          best_score := score;
          best_threshold := threshold
        end
      end
    done;
    
    j, !best_threshold, !best_score
    
  (* L1 version of VarianceSplit *)
  let find_l1_variance_split x y =
    let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
    
    let best_feature = ref 0 in
    let best_threshold = ref 0.0 in
    let best_score = ref Float.infinity in
    
    for j = 0 to d - 1 do
      let x_j = Tensor.select x ~dim:1 ~index:j in
      let min_val = Tensor.min x_j |> fst |> Tensor.item |> Float.to_float in
      let max_val = Tensor.max x_j |> fst |> Tensor.item |> Float.to_float in
      
      (* Try different thresholds *)
      let step = (max_val -. min_val) /. 10.0 in
      for t = 1 to 9 do
        let threshold = min_val +. step *. float_of_int t in
        
        let lt_mask = mask_lt x j threshold in
        let ge_mask = mask_ge x j threshold in
        
        let n_lt = Tensor.sum lt_mask |> Tensor.item |> Float.to_float in
        let n_ge = Tensor.sum ge_mask |> Tensor.item |> Float.to_float in
        
        if n_lt > 0.0 && n_ge > 0.0 then begin
          let l1_lt = 
            let y_lt = Tensor.masked_select y lt_mask in
            l1_deviation y_lt |> Tensor.item |> Float.to_float
          in
          
          let l1_ge = 
            let y_ge = Tensor.masked_select y ge_mask in
            l1_deviation y_ge |> Tensor.item |> Float.to_float
          in
          
          let p_lt = n_lt /. float_of_int n in
          let p_ge = n_ge /. float_of_int n in
          
          let score = p_lt *. l1_lt +. p_ge *. l1_ge in
          
          if score < !best_score then begin
            best_score := score;
            best_feature := j;
            best_threshold := threshold
          end
        end
      done
    done;
    
    !best_feature, !best_threshold, !best_score
    
  (* L1 version of MinimaxSplit *)
  let find_l1_minimax_split x y =
    let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
    
    let best_feature = ref 0 in
    let best_threshold = ref 0.0 in
    let best_score = ref Float.infinity in
    
    for j = 0 to d - 1 do
      let x_j = Tensor.select x ~dim:1 ~index:j in
      let min_val = Tensor.min x_j |> fst |> Tensor.item |> Float.to_float in
      let max_val = Tensor.max x_j |> fst |> Tensor.item |> Float.to_float in
      
      (* Try different thresholds *)
      let step = (max_val -. min_val) /. 10.0 in
      for t = 1 to 9 do
        let threshold = min_val +. step *. float_of_int t in
        
        let lt_mask = mask_lt x j threshold in
        let ge_mask = mask_ge x j threshold in
        
        let n_lt = Tensor.sum lt_mask |> Tensor.item |> Float.to_float in
        let n_ge = Tensor.sum ge_mask |> Tensor.item |> Float.to_float in
        
        if n_lt > 0.0 && n_ge > 0.0 then begin
          let l1_lt = 
            let y_lt = Tensor.masked_select y lt_mask in
            l1_deviation y_lt |> Tensor.item |> Float.to_float
          in
          
          let l1_ge = 
            let y_ge = Tensor.masked_select y ge_mask in
            l1_deviation y_ge |> Tensor.item |> Float.to_float
          in
          
          let p_lt = n_lt /. float_of_int n in
          let p_ge = n_ge /. float_of_int n in
          
          (* Maximize between the two L1 deviations *)
          let score = max (p_lt *. l1_lt) (p_ge *. l1_ge) in
          
          if score < !best_score then begin
            best_score := score;
            best_feature := j;
            best_threshold := threshold
          end
        end
      done
    done;
    
    !best_feature, !best_threshold, !best_score
  
  (* L1 version of CyclicMinimaxSplit *)
  let find_l1_cyclic_minimax_split x y j =
    let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
    
    let best_threshold = ref 0.0 in
    let best_score = ref Float.infinity in
    
    let x_j = Tensor.select x ~dim:1 ~index:j in
    let min_val = Tensor.min x_j |> fst |> Tensor.item |> Float.to_float in
    let max_val = Tensor.max x_j |> fst |> Tensor.item |> Float.to_float in
    
    (* Try different thresholds *)
    let step = (max_val -. min_val) /. 10.0 in
    for t = 1 to 9 do
      let threshold = min_val +. step *. float_of_int t in
      
      let lt_mask = mask_lt x j threshold in
      let ge_mask = mask_ge x j threshold in
      
      let n_lt = Tensor.sum lt_mask |> Tensor.item |> Float.to_float in
      let n_ge = Tensor.sum ge_mask |> Tensor.item |> Float.to_float in
      
      if n_lt > 0.0 && n_ge > 0.0 then begin
        let l1_lt = 
          let y_lt = Tensor.masked_select y lt_mask in
          l1_deviation y_lt |> Tensor.item |> Float.to_float
        in
        
        let l1_ge = 
          let y_ge = Tensor.masked_select y ge_mask in
          l1_deviation y_ge |> Tensor.item |> Float.to_float
        in
        
        let p_lt = n_lt /. float_of_int n in
        let p_ge = n_ge /. float_of_int n in
        
        (* Maximize between the two L1 deviations *)
        let score = max (p_lt *. l1_lt) (p_ge *. l1_ge) in
        
        if score < !best_score then begin
          best_score := score;
          best_threshold := threshold
        end
      end
    done;
    
    j, !best_threshold, !best_score
    
  (* Grow a decision tree recursively *)
  let rec grow_tree x y depth max_depth criterion node_id node k_mod_d =
    let x_filtered = Tensor.masked_select_nd x (Tensor.ones [Tensor.shape x |> List.hd]) in
    let y_filtered = Tensor.masked_select y (Tensor.ones [Tensor.shape y |> List.hd]) in
    
    (* Calculate stats for this node *)
    let samples = Tensor.shape y_filtered |> List.hd in
    let value = Tensor.mean y_filtered |> Tensor.item |> Float.to_float in
    let mse = 
      if samples > 0 then
        let preds = Tensor.full [samples] value in
        Tensor.(mean ((y_filtered - preds) ** (Scalar.f 2.0)))
        |> Tensor.item
        |> Float.to_float
      else 0.0
    in
    
    let node = create_node node.region value ~samples ~depth ~mse in
    
    if depth >= max_depth || not (is_splittable x_filtered y_filtered node.region) then
      node  (* Leaf node *)
    else
      (* Find the best split *)
      let feature, threshold, score = match criterion with
        | VarianceSplit -> find_variance_split x_filtered y_filtered
        | MinimaxSplit -> find_minimax_split x_filtered y_filtered
        | CyclicMinimaxSplit -> 
            let j = (k_mod_d + depth) mod (Tensor.shape x |> List.tl |> List.hd) in
            find_cyclic_minimax_split x_filtered y_filtered j
        | L1VarianceSplit -> find_l1_variance_split x_filtered y_filtered
        | L1MinimaxSplit -> find_l1_minimax_split x_filtered y_filtered
        | L1CyclicMinimaxSplit ->
            let j = (k_mod_d + depth) mod (Tensor.shape x |> List.tl |> List.hd) in
            find_l1_cyclic_minimax_split x_filtered y_filtered j
      in
      
      (* Create masks for the left and right children *)
      let lt_mask = mask_lt x_filtered feature threshold in
      let ge_mask = mask_ge x_filtered feature threshold in
      
      (* Filter data for left and right children *)
      let x_left = Tensor.masked_select_nd x_filtered lt_mask in
      let y_left = Tensor.masked_select y_filtered lt_mask in
      
      let x_right = Tensor.masked_select_nd x_filtered ge_mask in
      let y_right = Tensor.masked_select y_filtered ge_mask in
      
      (* Calculate child node stats *)
      let samples_left = Tensor.shape y_left |> List.hd in
      let samples_right = Tensor.shape y_right |> List.hd in
      
      let value_left = 
        if samples_left > 0 then Tensor.mean y_left |> Tensor.item |> Float.to_float 
        else node.value
      in
      
      let value_right = 
        if samples_right > 0 then Tensor.mean y_right |> Tensor.item |> Float.to_float
        else node.value
      in
      
      let mse_left = 
        if samples_left > 0 then
          let preds_left = Tensor.full [samples_left] value_left in
          Tensor.(mean ((y_left - preds_left) ** (Scalar.f 2.0)))
          |> Tensor.item
          |> Float.to_float
        else 0.0
      in
      
      let mse_right = 
        if samples_right > 0 then
          let preds_right = Tensor.full [samples_right] value_right in
          Tensor.(mean ((y_right - preds_right) ** (Scalar.f 2.0)))
          |> Tensor.item
          |> Float.to_float
        else 0.0
      in
      
      (* Create left and right regions *)
      let left_region = Array.copy node.region in
      left_region.(feature) <- (fst node.region.(feature), threshold);
      
      let right_region = Array.copy node.region in
      right_region.(feature) <- (threshold, snd node.region.(feature));
      
      (* Left child *)
      let left_node = create_node left_region value_left 
                        ~samples:samples_left ~depth:(depth + 1) ~mse:mse_left in
      let left_child = grow_tree x_left y_left (depth + 1) max_depth criterion (2 * node_id) left_node k_mod_d in
      
      (* Right child *)
      let right_node = create_node right_region value_right
                         ~samples:samples_right ~depth:(depth + 1) ~mse:mse_right in
      let right_child = grow_tree x_right y_right (depth + 1) max_depth criterion (2 * node_id + 1) right_node k_mod_d in
      
      (* Update the current node with the split information and children *)
      { node with
        feature = Some feature;
        threshold = Some threshold;
        left = Some left_child;
        right = Some right_child
      }
      
  (* Fit a decision tree on the data *)
  let fit x y max_depth criterion =
    let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
    
    (* Calculate the initial region bounds *)
    let region = Array.init d (fun j ->
      let x_j = Tensor.select x ~dim:1 ~index:j in
      let min_val = Tensor.min x_j |> fst |> Tensor.item |> Float.to_float in
      let max_val = Tensor.max x_j |> fst |> Tensor.item |> Float.to_float in
      (min_val, max_val)
    ) in
    
    (* Calculate root node stats *)
    let samples = n in
    let value = Tensor.mean y |> Tensor.item |> Float.to_float in
    let mse = 
      let preds = Tensor.full [samples] value in
      Tensor.(mean ((y - preds) ** (Scalar.f 2.0)))
      |> Tensor.item
      |> Float.to_float
    in
    
    let root = create_node region value ~samples ~depth:0 ~mse in
    let tree = grow_tree x y 0 max_depth criterion 1 root 0 in
    
    (* Calculate and print tree metrics for debugging *)
    let metrics = calc_node_metrics tree in
    Printf.printf "Tree metrics: leaves=%d, max_depth=%d, avg_depth=%.2f, mse=%.6f\n"
      metrics.n_leaves metrics.max_depth metrics.avg_depth metrics.mse;
    
    tree
    
  (* Predict a single sample using the tree *)
  let rec predict_sample tree x =
    match tree.feature, tree.threshold with
    | Some feature, Some threshold ->
        if x.(feature) < threshold then
          match tree.left with
          | Some left -> predict_sample left x
          | None -> tree.value
        else
          match tree.right with
          | Some right -> predict_sample right x
          | None -> tree.value
    | _ -> tree.value (* Leaf node *)
    
  (* Predict multiple samples *)
  let predict tree x =
    let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
    let predictions = Array.init n (fun i ->
      let sample = Array.init d (fun j ->
        Tensor.get x [i; j] |> Tensor.item |> Float.to_float
      ) in
      predict_sample tree sample
    ) in
    
    Tensor.of_float1 predictions
    
  (* Calculate the mean squared error *)
  let mse y_true y_pred =
    Tensor.(mean ((y_true - y_pred) ** (Scalar.f 2.0)))
    
  (* Calculate the mean absolute error *)
  let mae y_true y_pred =
    Tensor.(mean (abs (y_true - y_pred)))
    
  (* Calculate R^2 score *)
  let r2_score y_true y_pred =
    let ss_total = 
      let y_mean = Tensor.mean y_true in
      Tensor.(sum ((y_true - y_mean) ** (Scalar.f 2.0)))
    in
    let ss_residual = Tensor.(sum ((y_true - y_pred) ** (Scalar.f 2.0))) in
    Tensor.(1.0 - (ss_residual / ss_total))
    
  (* Calculate explained variance score *)
  let explained_variance_score y_true y_pred =
    let var_y = variance y_true in
    let var_residual = variance Tensor.(y_true - y_pred) in
    Tensor.(1.0 - (var_residual / var_y))
    
  (* Compute the total variation norm of a function g *)
  let tv_norm g =
    (* Assumes g is piecewise linear *)
    let n = Array.length g - 1 in
    let rec compute_tv i acc =
      if i >= n then acc
      else
        let diff = abs_float (g.(i+1) -. g.(i)) in
        compute_tv (i+1) (acc +. diff)
    in
    compute_tv 0 0.0
    
  (* Cross-validation utility *)
  let cross_validate x y folds max_depth criterion =
    let n = Tensor.shape x |> List.hd in
    let fold_size = n / folds in
    
    let scores = Array.make folds 0.0 in
    
    for i = 0 to folds - 1 do
      (* Calculate test indices *)
      let test_start = i * fold_size in
      let test_end = if i = folds - 1 then n else (i + 1) * fold_size in
      
      (* Split data into train and test *)
      let x_train = Tensor.cat [
        (if test_start > 0 then 
          Tensor.narrow x ~dim:0 ~start:0 ~length:test_start
        else Tensor.empty [0]);
        (if test_end < n then
          Tensor.narrow x ~dim:0 ~start:test_end ~length:(n - test_end)
        else Tensor.empty [0])
      ] ~dim:0 in
      
      let y_train = Tensor.cat [
        (if test_start > 0 then 
          Tensor.narrow y ~dim:0 ~start:0 ~length:test_start
        else Tensor.empty [0]);
        (if test_end < n then
          Tensor.narrow y ~dim:0 ~start:test_end ~length:(n - test_end)
        else Tensor.empty [0])
      ] ~dim:0 in
      
      let x_test = Tensor.narrow x ~dim:0 ~start:test_start ~length:(test_end - test_start) in
      let y_test = Tensor.narrow y ~dim:0 ~start:test_start ~length:(test_end - test_start) in
      
      (* Train model *)
      let model = fit x_train y_train max_depth criterion in
      
      (* Evaluate on test data *)
      let y_pred = predict model x_test in
      let test_mse = mse y_test y_pred |> Tensor.item |> Float.to_float in
      
      scores.(i) <- test_mse;
    done;
    
    (* Return mean and standard deviation of scores *)
    let mean_score = Array.fold_left (+.) 0.0 scores /. float_of_int folds in
    
    let std_score =
      let sum_squared_diff = Array.fold_left (fun acc s ->
        acc +. ((s -. mean_score) ** 2.0)
      ) 0.0 scores in
      sqrt (sum_squared_diff /. float_of_int folds)
    in
    
    mean_score, std_score
    
  (* Random forest *)
  module RandomForest = struct
    type forest = {
      trees: node array;
      criteria: split_criterion array;
      feature_importances: float array option;  (* Feature importances if calculated *)
    }
    
    (* Bootstrap sampling with replacement *)
    let bootstrap x y =
      let n = Tensor.shape x |> List.hd in
      let indices = Array.init n (fun _ -> Random.int n) in
      
      let x_bootstrap = Tensor.stack (
        Array.map (fun i -> Tensor.select x ~dim:0 ~index:i) indices
        |> Array.to_list
      ) ~dim:0 in
      
      let y_bootstrap = Tensor.stack (
        Array.map (fun i -> Tensor.select y ~dim:0 ~index:i) indices
        |> Array.to_list
      ) ~dim:0 in
      
      x_bootstrap, y_bootstrap
    
    (* Calculate feature importances based on node impurity decrease *)
    let calculate_feature_importances trees d =
      let importance_sum = Array.make d 0.0 in
      let importance_count = Array.make d 0 in
      
      (* Traverse each tree and accumulate importance *)
      Array.iter (fun tree ->
        let rec traverse node parent_samples parent_mse =
          match node.feature, node.left, node.right with
          | Some feature, Some left, Some right ->
              (* Calculate impurity decrease: parent_impurity - weighted_child_impurity *)
              let n_left = left.samples in
              let n_right = right.samples in
              
              let weight_left = float_of_int n_left /. float_of_int parent_samples in
              let weight_right = float_of_int n_right /. float_of_int parent_samples in
              
              let impurity_decrease = parent_mse -. (
                weight_left *. left.mse +. weight_right *. right.mse
              ) in
              
              (* Accumulate importance *)
              importance_sum.(feature) <- importance_sum.(feature) +. impurity_decrease;
              importance_count.(feature) <- importance_count.(feature) + 1;
              
              (* Recursively traverse children *)
              traverse left n_left left.mse;
              traverse right n_right right.mse;
          | _ -> ()  (* Leaf node - no contribution *)
        in
        
        traverse tree tree.samples tree.mse;
      ) trees;
      
      (* Normalize importances *)
      let importances = Array.mapi (fun i sum ->
        let count = importance_count.(i) in
        if count > 0 then sum /. float_of_int count else 0.0
      ) importance_sum in
      
      (* Normalize to sum to 1.0 *)
      let total = Array.fold_left (+.) 0.0 importances in
      if total > 0.0 then
        Array.map (fun imp -> imp /. total) importances
      else
        importances
        
    (* Fit a random forest with multiple splitting criteria *)
    let fit x y n_trees max_depth criteria =
      let n, d = Tensor.shape x |> function [n; d] -> (n, d) | _ -> failwith "Expected 2D tensor" in
      
      let trees = Array.init n_trees (fun i ->
        (* Bootstrap sampling *)
        let x_bootstrap, y_bootstrap = bootstrap x y in
        
        (* Select the splitting criterion for this tree *)
        let criterion = criteria.(i mod Array.length criteria) in
        
        (* Train the tree *)
        fit x_bootstrap y_bootstrap max_depth criterion
      ) in
      
      (* Calculate feature importances *)
      let feature_importances = calculate_feature_importances trees d in
      
      { 
        trees; 
        criteria;
        feature_importances = Some feature_importances;
      }
    
    (* Predict using the random forest (average predictions) *)
    let predict forest x =
      let n = Tensor.shape x |> List.hd in
      let predictions = Array.init n (fun i ->
        let sample = Tensor.select x ~dim:0 ~index:i in
        
        (* Get predictions from all trees *)
        let tree_preds = Array.map (fun tree ->
          let sample_arr = Array.init (Tensor.shape sample |> List.hd) (fun j ->
            Tensor.get sample [j] |> Tensor.item |> Float.to_float
          ) in
          predict_sample tree sample_arr
        ) forest.trees in
        
        (* Average the predictions *)
        Array.fold_left (+.) 0. tree_preds /. float_of_int (Array.length tree_preds)
      ) in
      
      Tensor.of_float1 predictions
  end

(* Evaluate model performance *)
let evaluate_model model predict_fn x_test y_test =
    let y_pred = predict_fn model x_test in
    let mse = DecisionTree.mse y_test y_pred |> Tensor.item |> Float.to_float in
    mse
    
(* Run experiment comparing different tree algorithms *)
let compare_tree_algorithms x y max_depth test_ratio =
    let (x_train, y_train), (x_test, y_test) = train_test_split x y test_ratio in
    
    (* Define algorithms to compare *)
    let algorithms = [
      "VarianceSplit", DecisionTree.VarianceSplit;
      "MinimaxSplit", DecisionTree.MinimaxSplit;
      "CyclicMinimaxSplit", DecisionTree.CyclicMinimaxSplit;
    ] in
    
    List.map (fun (name, criterion) ->
      (* Train model *)
      let model = DecisionTree.fit x_train y_train max_depth criterion in
      
      (* Evaluate performance *)
      let mse = evaluate_model model DecisionTree.predict x_test y_test in
      
      (name, mse)
    ) algorithms
    
(* Run experiment comparing different forest configurations *)
let compare_forest_algorithms x y max_depth n_trees test_ratio =
    let (x_train, y_train), (x_test, y_test) = train_test_split x y test_ratio in
    
    (* Define forest configurations *)
    let configurations = [
      "VarianceSplit Forest", [|DecisionTree.VarianceSplit|];
      "MinimaxSplit Forest", [|DecisionTree.MinimaxSplit|];
      "CyclicMinimaxSplit Forest", [|DecisionTree.CyclicMinimaxSplit|];
      "Mixed Forest", [|DecisionTree.VarianceSplit; DecisionTree.MinimaxSplit; DecisionTree.CyclicMinimaxSplit|];
    ] in
    
    List.map (fun (name, criteria) ->
      (* Train forest *)
      let forest = DecisionTree.RandomForest.fit x_train y_train n_trees max_depth criteria in
      
      (* Evaluate performance *)
      let mse = evaluate_model forest DecisionTree.RandomForest.predict x_test y_test in
      
      (name, mse)
    ) configurations
    
(* Compare empirical results with theoretical bounds *)
let compare_empirical_with_theoretical x y max_depths d =
    let (x_train, y_train), (x_test, y_test) = train_test_split x y 0.2 in
    let n_train = Tensor.shape x_train |> List.hd in
    
    (* Estimate the total variation norm *)
    let g_tv = 5.0 in (* This would be estimated in practice *)
    
    (* Get the range of y values *)
    let y_max = Tensor.max y_train |> fst |> Tensor.item |> Float.to_float in
    let y_min = Tensor.min y_train |> fst |> Tensor.item |> Float.to_float in
    let m = y_max -. y_min in
    
    List.map (fun k ->
      (* Train the cyclic minimax model *)
      let model = DecisionTree.fit x_train y_train k DecisionTree.CyclicMinimaxSplit in
      
      (* Evaluate performance *)
      let empirical_mse = evaluate_model model DecisionTree.predict x_test y_test in
      
      (* Calculate theoretical bound *)
      let epsilon = 0.1 in (* Arbitrary choice *)
      let theoretical_bound = Theory.empirical_cyclic_minimax_bound k d epsilon g_tv m n_train in
      
      (k, empirical_mse, theoretical_bound)
    ) max_depths

let generate_2d_function_data n =
    let x1 = Tensor.rand [n] in
    let x2 = Tensor.rand [n] in
    
    (* Compute the function values based on equation (30) *)
    let u1 = Tensor.((x1 - (Scalar.f 0.5)) * (Scalar.f 3.0)) in
    let u2 = Tensor.((x2 - (Scalar.f 0.5)) * (Scalar.f 3.0)) in
    
    let term1 = Tensor.((Scalar.f 2.0) - ((Scalar.f 2.1) * (u1 ** (Scalar.f 2.0))) + (u1 ** (Scalar.f 4.0)) / (Scalar.f 3.0)) in
    let term2 = Tensor.(u1 * u2) in
    let term3 = Tensor.(((Scalar.f (-4.0)) + ((Scalar.f 4.0) * (u2 ** (Scalar.f 2.0)))) * (u2 ** (Scalar.f 2.0))) in
    
    let y = Tensor.(term1 * (u1 ** (Scalar.f 2.0)) + term2 + term3) in
    
    (* Combine x1 and x2 into a single feature tensor *)
    let x = Tensor.stack [x1; x2] ~dim:1 in
    
    x, y
    
(* Evaluation with train-test split *)
let evaluate_model fit_fn predict_fn x y test_ratio =
    let n = Tensor.shape x |> List.hd in
    let n_test = int_of_float (float_of_int n *. test_ratio) in
    let n_train = n - n_test in
    
    (* Split the data *)
    let x_train = Tensor.narrow x ~dim:0 ~start:0 ~length:n_train in
    let y_train = Tensor.narrow y ~dim:0 ~start:0 ~length:n_train in
    
    let x_test = Tensor.narrow x ~dim:0 ~start:n_train ~length:n_test in
    let y_test = Tensor.narrow y ~dim:0 ~start:n_train ~length:n_test in
    
    (* Fit the model *)
    let model = fit_fn x_train y_train in
    
    (* Evaluate on test data *)
    let y_pred = predict_fn model x_test in
    let mse_val = DecisionTree.mse y_test y_pred |> Tensor.item |> Float.to_float in
    
    mse_val, model
    
(* Run experiment with multiple algorithms and report results *)
let compare_algorithms x y max_depth =
    let algos = [
      "VarianceSplit", DecisionTree.VarianceSplit;
      "MinimaxSplit", DecisionTree.MinimaxSplit;
      "CyclicMinimaxSplit", DecisionTree.CyclicMinimaxSplit;
      "L1VarianceSplit", DecisionTree.L1VarianceSplit;
      "L1MinimaxSplit", DecisionTree.L1MinimaxSplit;
      "L1CyclicMinimaxSplit", DecisionTree.L1CyclicMinimaxSplit;
    ] in
    
    List.map (fun (name, criterion) ->
      let mse_val, _ = evaluate_model 
        (fun x_train y_train -> DecisionTree.fit x_train y_train max_depth criterion)
        DecisionTree.predict
        x y 0.2 in
      name, mse_val
    ) algos
    
(* Compare random forest with various configurations *)
let compare_forest_configs x y max_depth n_trees =
    let configs = [
      "VarianceSplit Forest", [|DecisionTree.VarianceSplit|];
      "MinimaxSplit Forest", [|DecisionTree.MinimaxSplit|];
      "CyclicMinimaxSplit Forest", [|DecisionTree.CyclicMinimaxSplit|];
      "Mixed Forest", [|DecisionTree.VarianceSplit; DecisionTree.MinimaxSplit; DecisionTree.CyclicMinimaxSplit|];
    ] in
    
    List.map (fun (name, criteria) ->
      let mse_val, _ = evaluate_model
        (fun x_train y_train -> 
           DecisionTree.RandomForest.fit x_train y_train n_trees max_depth criteria)
        DecisionTree.RandomForest.predict
        x y 0.2 in
      name, mse_val
    ) configs
end

let variance_martingale_bound k =
    2.71 *. (2. ** ((-2. *. float_of_int k) /. 3.))
    
let simons_martingale_bound k =
    2. ** (-.float_of_int k)
    
let minimax_martingale_bound k =
    0.4 *. (2. ** ((-2. *. float_of_int k) /. 3.))
    
let median_martingale_bound k =
    2. ** (-.float_of_int k)
    
(* Non-uniform convergence rates *)
let non_uniform_martingale_bound k ~r ~c =
    c *. (r ** float_of_int k)
    
(* Asymptotic convergence rates *)
let asymptotic_minimax_bound k ~r ~c ~law_dependent =
    if law_dependent && r > 0.25 then
      c *. (r ** float_of_int k)
    else
      c *. (0.25 ** float_of_int k)
    
(* Cyclic minimax split *)
let cyclic_minimax_bound k d epsilon g_tv g_tv_range =
    (* Calculate k/d value for the floor function in the paper *)
    let k_over_d = float_of_int (k / d) in
    
    (* Compute the bound *)
    let delta = epsilon in
    let term1 = (1. +. delta) in
    let term2 = 2. *. (1. +. delta) *. (1. +. (1. /. delta)) in
    let term3 = 3. *. (2. ** (2. *. k_over_d /. 3.)) in
    let term4 = (1. +. (1. /. delta)) in
    let term5 = ((1. /. 3.) +. ((1. +. (1. /. delta)) /. 4.) ** (2. /. 3.)) in
    let term6 = 2. ** (-2. *. k_over_d /. 3.) in
    
    (term1 +. term2 /. term3) *. g_tv +. term4 *. term5 *. term6 *. g_tv_range
    
(* Empirical risk bound *)
let empirical_cyclic_minimax_bound k d epsilon g_tv m n =
    let k_over_d = float_of_int (k / d) in
    
    (* Compensation term *)
    let term1 = (1. +. (1. /. epsilon)) *. (2. ** (-2. *. k_over_d /. 3.)) in
    let term2 = (2. ** (float_of_int k +. 2.)) *. (m ** 2.) in
    let term3 = 3. *. float_of_int n in
    
    (* Base bound from cyclic_minimax_bound *)
    let base_bound = cyclic_minimax_bound k d epsilon g_tv g_tv in
    
    (* Complete bound *)
    term1 *. term2 /. term3 +. base_bound
    
(* Empirical risk oracle inequality *)
let oracle_inequality k d n g_star_norm =
    let k_over_d = float_of_int (k / d) in
    let delta = 2. ** (-2. *. k_over_d /. 3.) in
    
    (* Main terms *)
    let log_n = log (float_of_int n) /. log 2. in  (* log base 2 of n *)
    let c = 1.0 in  (* Constant from paper *)
    
    c *. (float_of_int n ** (-2.0 /. (3. *. float_of_int d +. 2.0))) *. 
    (g_star_norm +. (log_n ** 2.) *. log (float_of_int n *. float_of_int d))