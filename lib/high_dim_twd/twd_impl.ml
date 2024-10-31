open Torch

type feature_vector = Tensor.t
type sample_vector = Tensor.t
type diffusion_operator = Tensor.t
type hyperbolic_point = Tensor.t

type tree = {
  nodes: int array;
  edges: (int * int) array;
  weights: float array;
  parents: int array;
  subtree_leaves: int list array;
}

let embed_features features operator scale =
  let n = Array.length features in
  let embedded_points = Array.make n (Tensor.zeros [2]) in
  
  (* Use truncated SVD for large matrices *)
  let k = min (Tensor.size operator 0) 100 in
  let eigenvals, eigenvecs = Cache.get_eigendecomp operator in
  let eigenvals = Tensor.narrow eigenvals ~dim:0 ~start:0 ~length:k in
  let eigenvecs = Tensor.narrow eigenvecs ~dim:1 ~start:0 ~length:k in
  
  let t = Float.pow 2. (float_of_int scale) in
  
  for i = 0 to n - 1 do
    let feature = features.(i) in
    let scaled_eigenvals = Tensor.pow eigenvals t in
    let diffusion_coords = Tensor.mm 
      (Tensor.reshape feature [1; -1]) 
      (Tensor.mul eigenvecs (Tensor.reshape scaled_eigenvals [-1; 1])) in
    
    let scale_coord = Tensor.full [1; 1] (Float.pow 2. (float_of_int scale /. 2.)) in
    embedded_points.(i) <- Tensor.squeeze (Tensor.cat [diffusion_coords; scale_coord] ~dim:1)
  done;
  
  embedded_points

let compute_hd_lca point1 point2 =
  let dim = Tensor.size point1 0 in
  
  (* Get projection onto vertical axis following Proposition 1 *)
  let diff = Tensor.sub point1 point2 in
  let diff_front = Tensor.narrow diff ~dim:0 ~start:0 ~length:(dim-1) in
  let scale = Tensor.select point1 (dim-1) 0 in
  
  (* Compute projection value *)
  let proj_value = Tensor.norm diff_front in
  let proj_value = Tensor.div proj_value (Tensor.mul_scalar scale 2.0) in
  
  (* Construct LCA point *)
  let mean_front = Tensor.div (Tensor.add point1 point2) 2.0 in
  let mean_front = Tensor.narrow mean_front ~dim:0 ~start:0 ~length:(dim-1) in
  
  Tensor.cat [mean_front; Tensor.reshape proj_value [-1]] ~dim:0

let compute_multi_scale_hd_lca point1 point2 =
  Cache.get_hd_lca point1 point2

let construct_binary_tree_from_lca_vals lca_vals points =
  let n = Array.length points in
  let next_id = ref n in  (* Start node IDs after leaves *)
  
  (* Initialize tree components *)
  let nodes = Array.init (2*n) (fun i -> i) in
  let edges = ref [] in
  let weights = ref [] in
  let parents = Array.make (2*n) (-1) in
  let subtree_leaves = Array.make (2*n) [] in
  
  (* Initialize leaves *)
  for i = 0 to n-1 do
    subtree_leaves.(i) <- [i]
  done;
  
  (* Build tree bottom-up *)
  let active_nodes = ref (List.init n (fun i -> i)) in
  while List.length !active_nodes > 1 do
    (* Find closest pair *)
    let min_val = ref infinity in
    let min_i = ref 0 in
    let min_j = ref 0 in
    List.iter (fun i ->
      List.iter (fun j ->
        if i < j && lca_vals.(i).(j) < !min_val then begin
          min_val := lca_vals.(i).(j);
          min_i := i;
          min_j := j
        end
      ) !active_nodes
    ) !active_nodes;
    
    (* Create new internal node *)
    let new_node = !next_id in
    incr next_id;
    
    (* Update tree structure *)
    edges := (!min_i, new_node) :: (!min_j, new_node) :: !edges;
    weights := !min_val :: !min_val :: !weights;
    parents.(!min_i) <- new_node;
    parents.(!min_j) <- new_node;
    subtree_leaves.(new_node) <- subtree_leaves.(!min_i) @ subtree_leaves.(!min_j);
    
    (* Update active nodes *)
    active_nodes := List.filter (fun x -> x <> !min_i && x <> !min_j) !active_nodes;
    active_nodes := new_node :: !active_nodes
  done;
  
  {
    nodes;
    edges = Array.of_list (List.rev !edges);
    weights = Array.of_list (List.rev !weights);
    parents;
    subtree_leaves;
  }

let construct_binary_tree_parallel points =
  let n = Array.length points in
  let num_threads = 4 in
  let chunk_size = n / num_threads in
  
  (* Compute LCA values in parallel chunks *)
  let lca_vals = Array.make_matrix n n infinity in
  let workers = Array.init num_threads (fun thread_id ->
    let start_i = thread_id * chunk_size in
    let end_i = if thread_id = num_threads - 1 then n else (thread_id + 1) * chunk_size in
    
    Domain.spawn (fun () ->
      for i = start_i to end_i - 1 do
        for j = i + 1 to n - 1 do
          let lca = compute_multi_scale_hd_lca points.(i) points.(j) in
          let val_ = Tensor.get lca [|Tensor.size lca 0 - 1|] in
          lca_vals.(i).(j) <- val_;
          lca_vals.(j).(i) <- val_
        done
      done
    )
  ) in
  
  Array.iter Domain.join workers;
  construct_binary_tree_from_lca_vals lca_vals points

let construct_binary_tree = construct_binary_tree_parallel

let compute_twd_single sample1 sample2 tree =
  (* Normalize samples *)
  let normalize_sample s =
    let sum = Tensor.sum s ~dim:[0] ~keepdim:true in
    Tensor.div s sum
  in
  let s1 = normalize_sample sample1 in
  let s2 = normalize_sample sample2 in
  
  let total_distance = ref 0.0 in
  
  (* For each node in the tree *)
  Array.iteri (fun v _ ->
    if tree.parents.(v) >= 0 then begin  (* Skip root *)
      (* Get edge weight *)
      let edge_idx = ref 0 in
      Array.iteri (fun i (src, _) ->
        if src = v then edge_idx := i
      ) tree.edges;
      let alpha_v = tree.weights.(!edge_idx) in
      
      (* Sum differences in subtree *)
      let subtree_diff = ref 0.0 in
      List.iter (fun leaf ->
        let diff = Tensor.get s1 [|leaf|] -. Tensor.get s2 [|leaf|] in
        subtree_diff := !subtree_diff +. diff
      ) tree.subtree_leaves.(v);
      
      total_distance := !total_distance +. alpha_v *. abs_float !subtree_diff
    end
  ) tree.nodes;
  
  !total_distance

let compute_twd ?(use_sliced=false) sample1 sample2 tree =
  if not use_sliced then
    compute_twd_single sample1 sample2 tree
  else
    let max_scale = 3 in (* Default max scale *)
    let total_distance = ref 0.0 in
    
    (* For each scale *)
    for k = 0 to max_scale do
      (* Extract subtree for this scale *)
      let scale_tree = {
        nodes = tree.nodes;
        edges = tree.edges;
        weights = Array.map (fun w -> w *. Float.pow 2. (float_of_int k)) tree.weights;
        parents = tree.parents;
        subtree_leaves = tree.subtree_leaves;
      } in
      
      (* Compute TWD at this scale *)
      let scale_dist = compute_twd_single sample1 sample2 scale_tree in
      total_distance := !total_distance +. scale_dist
    done;
    
    !total_distance

let compute_twd_gpu ?(use_sliced=false) sample1 sample2 tree config =
  let s1 = Config.get_device_tensor config sample1 in
  let s2 = Config.get_device_tensor config sample2 in
  
  let result = compute_twd ~use_sliced s1 s2 tree in
  
  (* Move result back to CPU if needed *)
  match config.Config.device with
  | CPU -> result
  | GPU _ -> 
      let _ = Tensor.cpu s1 in
      let _ = Tensor.cpu s2 in
      result

let compute_scale_embedding_sparse features operator scale =
  let n = Array.length features in
  let embedded_points = Array.make n (Tensor.zeros [2]) in
  
  (* Use randomized SVD for large sparse matrices *)
  let k = min (Array.get operator.SparseOps.size 0) 100 in
  let q = k + 20 in (* Oversampling parameter *)
  
  (* Random projection matrix *)
  let omega = Tensor.randn [Array.get operator.SparseOps.size 1; q] in
  
  (* Power iteration *)
  let num_power_iter = 2 in
  let y = ref (SparseOps.sparse_mm operator (SparseOps.from_dense omega 0.)) in
  for _ = 1 to num_power_iter do
    y := SparseOps.sparse_mm operator !y
  done;
  
  (* QR decomposition of Y *)
  let q_mat = Tensor.qr (SparseOps.to_dense !y) |> fst in
  
  (* Form smaller matrix B = Q^T * A *)
  let b = Tensor.mm (Tensor.transpose q_mat ~dim0:0 ~dim1:1) 
    (SparseOps.to_dense operator) in
  
  (* SVD of B *)
  let u_tilde, s, _ = Tensor.svd b ~some:true in
  let u = Tensor.mm q_mat u_tilde in
  
  (* Truncate to k components *)
  let u_k = Tensor.narrow u ~dim:1 ~start:0 ~length:k in
  let s_k = Tensor.narrow s ~dim:0 ~start:0 ~length:k in
  
  (* Compute embeddings *)
  let t = Float.pow 2. (float_of_int scale) in
  
  for i = 0 to n - 1 do
    let feature = features.(i) in
    let scaled_s = Tensor.pow s_k t in
    let diffusion_coords = Tensor.mm 
      (Tensor.reshape feature [1; -1]) 
      (Tensor.mul u_k (Tensor.reshape scaled_s [-1; 1])) in
    
    let scale_coord = Tensor.full [1; 1] (Float.pow 2. (float_of_int scale /. 2.)) in
    embedded_points.(i) <- Tensor.squeeze (Tensor.cat [diffusion_coords; scale_coord] ~dim:1)
  done;
  
  embedded_points