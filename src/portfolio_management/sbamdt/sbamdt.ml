open Torch

type feature_space = {
  structured: Tensor.t;   (* s ∈ M *)
  unstructured: Tensor.t; (* x ∈ X ⊆ Rp *)
  dim_structured: int;
  dim_unstructured: int;
}

type model_variant = 
  | S2BAMDT of float  (* Single alpha parameter *)
  | SkBAMDT of float array (* Multiple fixed alphas *)

type model_params = {
  num_trees: int;
  gamma: float;        (* Tree split probability control *)
  delta: float;        (* Tree depth penalty *)
  alpha_mu: float;     (* Inverse-gamma shape for leaf params *)
  beta_mu: float;      (* Inverse-gamma scale for leaf params *)
  sigma_sq: float;     (* Residual variance *)
  sigma_mu_sq: float;  (* Leaf parameter variance *)
  variant: model_variant;
  alpha_g: float;      (* Gamma shape for alpha prior *)
  beta_g: float;       (* Gamma rate for alpha prior *)
  nu: float;          (* Degrees of freedom for sigma_sq prior *)
  lambda: float;      (* Scale parameter for sigma_sq prior *)
  k_prior_counts: float array; (* Prior counts for Sk-BAMDT *)
}

type split_type = 
  | UnivariateRule of {
      feature_idx: int;
      threshold: float;
    }
  | MultivariateRule of {
      reference_points: Tensor.t;
      left_knots: Tensor.t;
      right_knots: Tensor.t;
    }

type decision_rule = {
  split: split_type;
  prob_type: [`Hard | `Soft of float]; (* alpha parameter *)
  normalization_const: float;
}

type tree_node =
  | Terminal of {
      indices: int array;
      parameter: float option;
    }
  | Internal of {
      left: tree;
      right: tree;
      rule: decision_rule;
      indices: int array;
    }
and tree = {
  node: tree_node;
  depth: int;
}

type model = {
  trees: tree list;
  params: model_params;
  feature_space: feature_space;
  observed_data: Tensor.t option;
}

(* Random Number Generation Utilities *)
module Random = struct
  let gaussian () =
    let u1 = Random.float 1.0 in
    let u2 = Random.float 1.0 in
    sqrt (-2.0 *. log u1) *. cos (2.0 *. Float.pi *. u2)

  let gamma shape scale =
    if shape < 1.0 then
      let u = Random.float 1.0 in
      (gamma (shape +. 1.0) scale) *. (u ** (1.0 /. shape))
    else
      let d = shape -. 1.0 /. 3.0 in
      let c = 1.0 /. sqrt (9.0 *. d) in
      let rec loop () =
        let x = gaussian () in
        let v = (1.0 +. c *. x) ** 3.0 in
        if v <= 0.0 then loop ()
        else
          let u = Random.float 1.0 in
          if log u < 0.5 *. x *. x +. d *. (1.0 -. v +. log v) then
            d *. v *. scale
          else loop ()
      in loop ()

  let inverse_gamma shape scale =
    1.0 /. (gamma shape (1.0 /. scale))

  let dirichlet alphas =
    let samples = Array.map (fun a -> gamma a 1.0) alphas in
    let sum = Array.fold_left (+.) 0.0 samples in
    Array.map (fun x -> x /. sum) samples
end

(* Tensor Operations *)
module TensorOps = struct
  open Types

  let gather tensor indices =
    let n = Array.length indices in
    let dim = Tensor.size tensor 1 in
    let result = Tensor.zeros [n; dim] in
    for i = 0 to n - 1 do
      let row = Tensor.narrow tensor 0 indices.(i) 1 in
      Tensor.copy_ (Tensor.narrow result 0 i 1) row
    done;
    result

  let compute_distances points refs =
    let n = Tensor.size points 0 in
    let m = Tensor.size refs 0 in
    let dists = Tensor.zeros [n; m] in
    for i = 0 to n - 1 do
      let pi = Tensor.narrow points 0 i 1 in
      for j = 0 to m - 1 do
        let rj = Tensor.narrow refs 0 j 1 in
        let diff = Tensor.sub pi rj in
        let dist = Tensor.norm diff Tensor.Norm.Float in
        Tensor.set dists [|i; j|] (Tensor.float_value dist)
      done
    done;
    dists
end

(* Laplacian Operations *)
module LaplacianOps = struct
  open Types

  let compute_similarity_matrix points sigma =
    let n = Tensor.size points 0 in
    let sim = Tensor.zeros [n; n] in
    for i = 0 to n-1 do
      for j = i+1 to n-1 do
        let pi = Tensor.narrow points 0 i 1 in
        let pj = Tensor.narrow points 0 j 1 in
        let dist = Tensor.(norm (sub pi pj) Norm.Float |> float_value) in
        let sim_val = exp (-. dist *. dist /. (2.0 *. sigma *. sigma)) in
        Tensor.(
          set sim [|i; j|] sim_val;
          set sim [|j; i|] sim_val
        )
      done
    done;
    sim

  let compute_normalized_laplacian sim =
    let n = Tensor.size sim 0 in
    let degree = Tensor.sum sim 1 in
    let d_sqrt_inv = Tensor.pow degree (-0.5) in
    let d_mat = Tensor.diag d_sqrt_inv in
    let l = Tensor.matmul (Tensor.matmul d_mat sim) d_mat in
    Tensor.sub (Tensor.eye n) l

  let compute_embedding points k sigma =
    (* Compute Laplacian eigenvectors *)
    let sim = compute_similarity_matrix points sigma in
    let lap = compute_normalized_laplacian sim in
    let eigvals, eigvecs = Tensor.symeig lap ~eigenvectors:true in
    
    (* Take k smallest non-zero eigenvectors *)
    let n_dims = min k (Tensor.size eigvecs 1) in
    Tensor.narrow eigvecs 1 1 n_dims

  let manifold_distance points embedding sigma =
    let n = Tensor.size points 0 in
    let embedded = compute_embedding points sigma 10 in
    let dists = Tensor.zeros [n; n] in
    
    for i = 0 to n-1 do
      for j = i+1 to n-1 do
        let ei = Tensor.narrow embedded 0 i 1 in
        let ej = Tensor.narrow embedded 0 j 1 in
        let dist = Tensor.(norm (sub ei ej) Norm.Float |> float_value) in
        Tensor.(
          set dists [|i; j|] dist;
          set dists [|j; i|] dist
        )
      done
    done;
    dists
end

(* Reference Point Selection and Management *)
module ReferencePoints = struct
  open Types
  
  let select_kmeans_refs points n_refs max_iters =
    let n = Tensor.size points 0 in
    let dim = Tensor.size points 1 in
    let refs = Tensor.zeros [n_refs; dim] in
    
    (* Initialize with random points *)
    for i = 0 to n_refs - 1 do
      let idx = Random.int n in
      let point = Tensor.narrow points 0 idx 1 in
      Tensor.copy_ (Tensor.narrow refs 0 i 1) point
    done;
    
    let assignments = Array.make n 0 in
    let changed = ref true in
    let iter = ref 0 in
    
    while !changed && !iter < max_iters do
      changed := false;
      
      (* Assign points to nearest reference *)
      for i = 0 to n - 1 do
        let point = Tensor.narrow points 0 i 1 in
        let min_dist = ref Float.infinity in
        let min_idx = ref 0 in
        
        for j = 0 to n_refs - 1 do
          let ref_point = Tensor.narrow refs 0 j 1 in
          let dist = Tensor.(norm (sub point ref_point) Norm.Float |> float_value) in
          if dist < !min_dist then begin
            min_dist := dist;
            min_idx := j
          end
        done;
        
        if assignments.(i) <> !min_idx then begin
          assignments.(i) <- !min_idx;
          changed := true
        end
      done;
      
      (* Update reference points *)
      if !changed then begin
        Tensor.zero_ refs;
        let counts = Array.make n_refs 0 in
        
        for i = 0 to n - 1 do
          let cluster = assignments.(i) in
          let point = Tensor.narrow points 0 i 1 in
          Tensor.(add_ (narrow refs 0 cluster 1) point);
          counts.(cluster) <- counts.(cluster) + 1
        done;
        
        for i = 0 to n_refs - 1 do
          if counts.(i) > 0 then
            Tensor.(div_ (narrow refs 0 i 1) (float counts.(i)))
        done
      end;
      
      incr iter
    done;
    refs

  let bipartition_refs refs embedding =
    let n = Tensor.size refs 0 in
    let min_cut_score = ref Float.infinity in
    let best_partition = ref (Array.make n true) in
    
    (* Try multiple random initializations *)
    for _ = 1 to 10 do
      let partition = Array.init n (fun _ -> Random.bool ()) in
      let mut_partition = Array.copy partition in
      let improved = ref true in
      
      while !improved do
        improved := false;
        for i = 0 to n - 1 do
          mut_partition.(i) <- not mut_partition.(i);
          let score = compute_cut_score mut_partition refs embedding in
          
          if score < !min_cut_score then begin
            min_cut_score := score;
            best_partition := Array.copy mut_partition;
            improved := true
          end else
            mut_partition.(i) <- not mut_partition.(i)
        done
      done
    done;
    
    partition_refs refs !best_partition
end

(* Decision Rules and Split Functions *)
module DecisionRules = struct
  open Types

  type split_prob = {
    left_prob: float;
    right_prob: float;
  }

  let create_univariate_rule features indices =
    let feat_idx = Random.int features.dim_unstructured in
    let feat_data = TensorOps.gather features.unstructured indices in
    
    (* Compute split threshold *)
    let min_val = Tensor.min feat_data |> Tensor.float_value in
    let max_val = Tensor.max feat_data |> Tensor.float_value in
    let threshold = min_val +. Random.float (max_val -. min_val) in
    
    UnivariateRule {
      feature_idx = feat_idx;
      threshold = threshold
    }

  let create_multivariate_rule features indices sigma =
    let point_data = TensorOps.gather features.structured indices in
    let embedding = LaplacianOps.compute_embedding point_data 10 sigma in
    
    (* Select and partition reference points *)
    let n_refs = min 20 (Array.length indices) in
    let refs = ReferencePoints.select_kmeans_refs point_data n_refs 10 in
    let left_refs, right_refs = ReferencePoints.bipartition_refs refs embedding in
    
    MultivariateRule {
      reference_points = refs;
      left_knots = left_refs;
      right_knots = right_refs;
    }

  let create_decision_rule params features indices =
    let use_multivariate = 
      Random.float 1.0 < float features.dim_structured /. 
        float (features.dim_structured + features.dim_unstructured) 
    in
    
    let split = 
      if use_multivariate then
        create_multivariate_rule features indices 1.0
      else
        create_univariate_rule features indices
    in
    
    let alpha = match params.variant with
      | S2BAMDT alpha -> alpha
      | SkBAMDT alphas -> 
          alphas.(Random.int (Array.length alphas))
    in
    
    {
      split;
      prob_type = if Random.float 1.0 < 0.5 then `Hard else `Soft alpha;
      normalization_const = 1.0;  (* Will be updated based on data *)
    }

  let compute_split_probability rule point =
    match (rule.split, rule.prob_type) with
    | UnivariateRule {feature_idx; threshold}, prob_type ->
        let value = Tensor.get point feature_idx in
        begin match prob_type with
        | `Hard -> if value <= threshold then 1.0 else 0.0
        | `Soft alpha ->
            let diff = (value -. threshold) /. rule.normalization_const in
            1.0 /. (1.0 +. exp (alpha *. diff))
        end
    | MultivariateRule {left_knots; right_knots; _}, prob_type ->
        let left_dist = TensorOps.compute_min_distance point left_knots in
        let right_dist = TensorOps.compute_min_distance point right_knots in
        begin match prob_type with
        | `Hard -> 
            if left_dist <= right_dist then 1.0 else 0.0
        | `Soft alpha ->
            let diff = (right_dist -. left_dist) /. rule.normalization_const in
            1.0 /. (1.0 +. exp (alpha *. diff))
        end

  let update_normalization_constant rule data =
    match rule.split with
    | UnivariateRule {feature_idx; threshold} ->
        let values = Tensor.select data 1 feature_idx in
        let diffs = Tensor.abs (Tensor.sub values (Tensor.float threshold)) in
        let max_diff = Tensor.max diffs |> Tensor.float_value in
        {rule with normalization_const = max_diff}
    | MultivariateRule {left_knots; right_knots; _} ->
        let n = Tensor.size data 0 in
        let max_dist = ref 0.0 in
        
        for i = 0 to n - 1 do
          let point = Tensor.narrow data 0 i 1 in
          let left_dist = TensorOps.compute_min_distance point left_knots in
          let right_dist = TensorOps.compute_min_distance point right_knots in
          let dist_diff = abs_float (right_dist -. left_dist) in
          if dist_diff > !max_dist then max_dist := dist_diff
        done;
        {rule with normalization_const = !max_dist}
end

(* Tree Node Operations *)
module TreeOps = struct
  open Types

  let rec create_tree depth max_depth params features indices =
    if depth >= max_depth || 
       Random.float 1.0 > params.gamma /. (1. +. float_of_int depth) ** params.delta then
      {
        node = Terminal {
          indices;
          parameter = None
        };
        depth
      }
    else
      let rule = DecisionRules.create_decision_rule params features indices in
      let left_indices, right_indices = 
        partition_data indices rule features in
      {
        node = Internal {
          left = create_tree (depth + 1) max_depth params features left_indices;
          right = create_tree (depth + 1) max_depth params features right_indices;
          rule;
          indices
        };
        depth
      }

    let partition_data indices rule features =
    let data = match rule.split with
      | UnivariateRule {feature_idx; _} ->
          TensorOps.gather features.unstructured indices
      | MultivariateRule _ ->
          TensorOps.gather features.structured indices
    in
    
    let n = Array.length indices in
    let left_indices = ref [] in
    let right_indices = ref [] in
    
    for i = 0 to n - 1 do
      let point = Tensor.narrow data 0 i 1 in
      let prob = DecisionRules.compute_split_probability rule point in
      
      let goes_left = match rule.prob_type with
        | `Hard -> prob > 0.5
        | `Soft _ -> Random.float 1.0 < prob
      in
      
      if goes_left then
        left_indices := indices.(i) :: !left_indices
      else
        right_indices := indices.(i) :: !right_indices
    done;
    
    (Array.of_list !left_indices, Array.of_list !right_indices)

  let find_leaf_paths tree =
    let rec find node path acc =
      match node with
      | Terminal _ -> (List.rev path) :: acc
      | Internal {left; right; _} ->
          let left_paths = find left.node (0 :: path) acc in
          find right.node (1 :: path) left_paths
    in
    find tree.node [] []

  let find_leaf_parent_paths tree =
    let rec find node path acc =
      match node with
      | Terminal _ -> acc
      | Internal {left; right; _} ->
          let acc' = match (left.node, right.node) with
            | (Terminal _, Terminal _) -> (List.rev path) :: acc
            | _ -> acc
          in
          let left_paths = find left.node (0 :: path) acc' in
          find right.node (1 :: path) left_paths
    in
    find tree.node [] []

  let find_internal_paths tree =
    let rec find node path acc =
      match node with
      | Terminal _ -> acc
      | Internal {left; right; _} ->
          let acc' = (List.rev path) :: acc in
          let left_paths = find left.node (0 :: path) acc' in
          find right.node (1 :: path) left_paths
    in
    find tree.node [] []

  let rec get_node_at_path tree path =
    match (tree.node, path) with
    | node, [] -> Some node
    | Internal {left; right; _}, dir :: rest ->
        get_node_at_path (if dir = 0 then left else right) rest
    | Terminal _, _ -> None

  let rec update_node_at_path tree path new_node =
    match (tree.node, path) with
    | _, [] -> {tree with node = new_node}
    | Internal {left; right; rule; indices}, dir :: rest ->
        let updated = 
          if dir = 0 then
            {tree with node = 
              Internal {
                left = update_node_at_path left rest new_node;
                right;
                rule;
                indices
              }}
          else
            {tree with node = 
              Internal {
                left;
                right = update_node_at_path right rest new_node;
                rule;
                indices
              }}
        in
        updated
    | Terminal _, _ -> tree

  let grow_at_path tree path params =
    match get_node_at_path tree path with
    | Some (Terminal {indices; _}) ->
        let new_rule = DecisionRules.create_decision_rule 
          params tree.feature_space indices in
        let left_indices, right_indices = 
          partition_data indices new_rule tree.feature_space in
        let new_node = Internal {
          left = {
            node = Terminal {indices = left_indices; parameter = None};
            depth = tree.depth + 1
          };
          right = {
            node = Terminal {indices = right_indices; parameter = None};
            depth = tree.depth + 1
          };
          rule = new_rule;
          indices
        } in
        update_node_at_path tree path new_node
    | _ -> tree

  let prune_at_path tree path =
    match get_node_at_path tree path with
    | Some (Internal {indices; _}) ->
        let new_node = Terminal {indices; parameter = None} in
        update_node_at_path tree path new_node
    | _ -> tree

  let change_rule_at_path tree path new_rule =
    match get_node_at_path tree path with
    | Some (Internal {left; right; indices; _}) ->
        let new_node = Internal {left; right; rule = new_rule; indices} in
        update_node_at_path tree path new_node
    | _ -> tree

  let predict_tree tree data =
    let n = Tensor.size data 0 in
    let predictions = Tensor.zeros [n] in
    
    let rec traverse node point =
      match node with
      | Terminal {parameter = Some p; _} -> p
      | Terminal {parameter = None; _} -> 0.0
      | Internal {left; right; rule; _} ->
          let prob = DecisionRules.compute_split_probability rule point in
          let left_pred = traverse left.node point in
          let right_pred = traverse right.node point in
          prob *. left_pred +. (1.0 -. prob) *. right_pred
    in
    
    for i = 0 to n - 1 do
      let point = Tensor.narrow data 0 i 1 in
      let pred = traverse tree.node point in
      Tensor.set predictions [|i|] pred
    done;
    predictions
end

(* MCMC Sampling *)
module MCMC = struct
  open Types

  type proposal = 
    | Grow of {depth: int; target_node: int list}
    | Prune of {target_node: int list}
    | Change of {node_path: int list; new_rule: decision_rule}

  let compute_likelihood tree data params =
    let predictions = TreeOps.predict_tree tree data in
    let n = Tensor.size data 0 in
    
    let residuals = Tensor.sub data predictions in
    let squared_error = Tensor.sum (Tensor.pow residuals 2.0) in
    
    -0.5 *. (float_of_int n) *. log (2.0 *. Float.pi *. params.sigma_sq) -.
    (1.0 /. (2.0 *. params.sigma_sq)) *. Tensor.float_value squared_error

  let compute_tree_prior tree params =
    let rec node_prior node depth =
      match node with
      | Terminal _ -> 0.0
      | Internal {left; right; _} ->
          let split_prob = 
            params.gamma /. (1. +. float_of_int depth) ** params.delta in
          log split_prob +.
          node_prior left.node (depth + 1) +.
          node_prior right.node (depth + 1)
    in
    node_prior tree.node 0

  let sample_proposal tree =
    let p = Random.float 1.0 in
    if p < 0.4 then
      (* Grow proposal *)
      let leaf_paths = TreeOps.find_leaf_paths tree in
      let target = List.nth leaf_paths (Random.int (List.length leaf_paths)) in
      Grow {depth = tree.depth; target_node = target}
    else if p < 0.8 then
      (* Prune proposal *)
      let parent_paths = TreeOps.find_leaf_parent_paths tree in
      let target = List.nth parent_paths (Random.int (List.length parent_paths)) in
      Prune {target_node = target}
    else
      (* Change proposal *)
      let internal_paths = TreeOps.find_internal_paths tree in
      let path = List.nth internal_paths (Random.int (List.length internal_paths)) in
      let new_rule = DecisionRules.create_decision_rule 
        tree.params tree.feature_space [] in
      Change {node_path = path; new_rule}

  let metropolis_hastings_step tree data params =
    let proposal = sample_proposal tree in
    let proposed_tree = match proposal with
      | Grow {target_node; _} ->
          TreeOps.grow_at_path tree target_node params
      | Prune {target_node} ->
          TreeOps.prune_at_path tree target_node
      | Change {node_path; new_rule} ->
          TreeOps.change_rule_at_path tree node_path new_rule
    in
    
    let current_likelihood = compute_likelihood tree data params in
    let proposed_likelihood = compute_likelihood proposed_tree data params in
    
    let current_prior = compute_tree_prior tree params in
    let proposed_prior = compute_tree_prior proposed_tree params in
    
    let log_ratio = 
      (proposed_likelihood +. proposed_prior) -.
      (current_likelihood +. current_prior)
    in
    
    if log (Random.float 1.0) < log_ratio then
      proposed_tree
    else
      tree
end

(* Parameter Updates *)
module ParameterUpdates = struct
  let update_sigma_sq model data =
    let n = Tensor.size data 0 in
    let predictions = List.map (fun tree -> 
      TreeOps.predict_tree tree data) model.trees in
    let total_pred = List.fold_left Tensor.add 
      (Tensor.zeros [n]) predictions in
    let residuals = Tensor.sub data total_pred in
    
    let shape = (float_of_int n +. model.params.nu) /. 2.0 in
    let scale = (Tensor.sum (Tensor.pow residuals 2.0) |> Tensor.float_value +.
                 model.params.nu *. model.params.lambda) /. 2.0 in
    
    Random.inverse_gamma shape scale

  let collect_leaf_parameters trees =
    let params = ref [] in
    let rec collect node =
      match node with
      | Terminal {parameter = Some p; _} -> 
          params := p :: !params
      | Terminal {parameter = None; _} -> ()
      | Internal {left; right; _} ->
          collect left.node;
          collect right.node
    in
    List.iter (fun tree -> collect tree.node) trees;
    !params

  let update_sigma_mu_sq model =
    let leaf_params = collect_leaf_parameters model.trees in
    let n = List.length leaf_params in
    let sum_squares = 
      List.fold_left (fun acc p -> acc +. p *. p) 0.0 leaf_params in
    
    let shape = model.params.alpha_mu +. float_of_int n /. 2.0 in
    let scale = model.params.beta_mu +. sum_squares /. 2.0 in
    Random.inverse_gamma shape scale

  let update_alpha model data =
    match model.params.variant with
    | S2BAMDT _ ->
        (* Update single alpha parameter for S2-BAMDT *)
        let count_soft_decisions tree =
          let count = ref 0 in
          let rec count_node node =
            match node with
            | Terminal _ -> ()
            | Internal {left; right; rule; _} ->
                begin match rule.prob_type with
                | `Soft _ -> incr count
                | `Hard -> ()
                end;
                count_node left.node;
                count_node right.node
          in
          count_node tree.node;
          !count
        in
        
        let total_soft = List.fold_left (fun acc tree ->
          acc + count_soft_decisions tree
        ) 0 model.trees in
        
        let shape = model.params.alpha_g +. float_of_int total_soft in
        let rate = model.params.beta_g +. 
          List.fold_left (fun acc tree ->
            acc +. compute_soft_likelihood tree data
          ) 0.0 model.trees in
        
        Random.gamma shape (1.0 /. rate)
        
    | SkBAMDT k_levels ->
        (* Update categorical probabilities for Sk-BAMDT *)
        let counts = Array.make (Array.length k_levels) 0 in
        List.iter (fun tree ->
          let rec count_node node =
            match node with
            | Terminal _ -> ()
            | Internal {left; right; rule; _} ->
                begin match rule.prob_type with
                | `Soft alpha ->
                    let idx = find_alpha_index alpha k_levels in
                    counts.(idx) <- counts.(idx) + 1
                | `Hard -> ()
                end;
                count_node left.node;
                count_node right.node
          in
          count_node tree.node
        ) model.trees;
        
        let posterior_counts = Array.map2 (+.) 
          (Array.map float_of_int counts) 
          model.params.k_prior_counts in
        Random.dirichlet posterior_counts

  let update_leaf_parameters tree data model_params =
    let rec update node =
      match node with
      | Terminal {indices; _} ->
          let subset_data = TensorOps.gather data indices in
          let mean = Tensor.mean subset_data 0 |> Tensor.float_value in
          let std = sqrt model_params.sigma_mu_sq in
          let param = mean +. Random.gaussian () *. std in
          Terminal {indices; parameter = Some param}
          
      | Internal {left; right; rule; indices} ->
          Internal {
            left = {left with node = update left.node};
            right = {right with node = update right.node};
            rule;
            indices
          }
    in
    {tree with node = update tree.node}

  let compute_soft_likelihood tree data =
    let rec compute_node node acc =
      match node with
      | Terminal _ -> acc
      | Internal {left; right; rule; _} ->
          let acc' = match rule.prob_type with
            | `Soft alpha ->
                let predictions = TreeOps.predict_tree tree data in
                let residuals = Tensor.sub data predictions in
                acc -. alpha *. 
                  (Tensor.sum (Tensor.pow residuals 2.0) |> Tensor.float_value)
            | `Hard -> acc
          in
          compute_node left.node (compute_node right.node acc')
    in
    compute_node tree.node 0.0

  let find_alpha_index alpha k_levels =
    let rec find idx =
      if idx >= Array.length k_levels then 0  (* Default to first level *)
      else if abs_float (k_levels.(idx) -. alpha) < 1e-6 then idx
      else find (idx + 1)
    in
    find 0
end