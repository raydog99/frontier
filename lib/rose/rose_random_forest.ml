type forest_params = {
  max_depth: int;
  mtry: int;
  min_node_size: int;
  n_trees: int;
  sample_fraction: float;
}

let create_rose_forest data nuisance theta params =
  let n = Array.length data in
  let n_samples = int_of_float (float_of_int n *. params.sample_fraction) in
  
  let trees = Array.init params.n_trees (fun _ ->
    (* Bootstrap sample *)
    let sample_indices = Array.init n_samples (fun _ -> Random.int n) in
    let sample_data = Array.map (fun i -> data.(i)) sample_indices in
    let sample_nuisance = Array.map (fun i -> nuisance.(i)) sample_indices in
    
    (* Split data into training and evaluation sets *)
    let n_split = n_samples / 2 in
    let train_indices = Array.sub sample_indices 0 n_split in
    let eval_indices = Array.sub sample_indices n_split (n_samples - n_split) in
    
    (* Build tree and compute weights *)
    let tree = RoseTree.build_tree 
      sample_data sample_nuisance theta train_indices 
      params.max_depth params.mtry params.min_node_size in
    
    (* Compute tau statistics *)
    let tau1 = Array.make n 0.0 in
    let tau2 = Array.make n 0.0 in
    
    Array.iter (fun i ->
      let weights = RoseTree.compute_node_weights tree data.(i).z in
      let (psi, d_theta) = Models.compute_influence 
        Models.PartiallyLinear data.(i) nuisance.(i) theta in
      tau1.(i) <- weights *. psi *. psi;
      tau2.(i) <- weights *. d_theta
    ) eval_indices;
    
    { tree with tau1; tau2 }
  ) in
  
  { trees = trees; n_trees = params.n_trees; folds = [||] }

let compute_optimal_weights forest data =
  let n = Array.length data in
  let n_trees = forest.n_trees in
  let weights = Array.make n 0.0 in
  
  for i = 0 to n - 1 do
    let tau1_sum = ref 0.0 in
    let tau2_sum = ref 0.0 in
    let max_tau1 = ref neg_infinity in
    let max_tau2 = ref neg_infinity in
    
    (* Find maximum values for numerical stability *)
    for b = 0 to n_trees - 1 do
      max_tau1 := max !max_tau1 forest.trees.(b).tau1.(i);
      max_tau2 := max !max_tau2 forest.trees.(b).tau2.(i)
    done;
    
    (* Compute sums with shifted values *)
    for b = 0 to n_trees - 1 do
      tau1_sum := !tau1_sum +. exp(forest.trees.(b).tau1.(i) -. !max_tau1);
      tau2_sum := !tau2_sum +. exp(forest.trees.(b).tau2.(i) -. !max_tau2)
    done;
    
    weights.(i) <- exp(!max_tau2 +. log !tau2_sum -. (!max_tau1 +. log !tau1_sum))
  done;
  
  (* Normalize weights *)
  let weight_sum = Array.fold_left (+.) 0.0 weights in
  Array.map (fun w -> w /. weight_sum) weights

let estimate data forest alpha =
  let weights = compute_optimal_weights forest data in
  let n = Array.length data in
  
  (* Solve estimating equations *)
  let theta_hat = Estimator.solve_estimating_equations data weights [||] 0.0 100 1e-6 in
  
  (* Compute variance estimate *)
  let psi = Array.make n 0.0 in
  let d_theta = Array.make n 0.0 in
  Array.iteri (fun i obs ->
    let (p, d) = Models.compute_influence 
      Models.PartiallyLinear obs [||] theta_hat in
    psi.(i) <- p;
    d_theta.(i) <- d
  ) data;
  
  let variance = Estimator.compute_sandwich_variance data weights psi d_theta in
  let ci = Estimator.compute_confidence_interval theta_hat variance alpha in
  
  { theta_hat; sigma_hat = sqrt variance; confidence_interval = ci }