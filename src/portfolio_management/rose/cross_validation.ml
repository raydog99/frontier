open Types

type cv_params = {
  max_depth: int;
  mtry: int;
  min_node_size: int;
  n_trees: int;
}

type cv_result = {
  params: cv_params;
  score: float;
}

let compute_cv_score data weights nuisance_estimates theta =
  let n = Array.length data in
  let psi = Array.make n 0.0 in
  let d_theta = Array.make n 0.0 in
  
  Array.iteri (fun i obs ->
    let (p, d) = Models.compute_influence 
      Models.PartiallyLinear obs nuisance_estimates.(i) theta in
    psi.(i) <- p;
    d_theta.(i) <- d
  ) data;
  
  let numerator = ref 0.0 in
  let denominator = ref 0.0 in
  
  Array.iteri (fun i _ ->
    numerator := !numerator +. weights.(i) *. d_theta.(i);
    denominator := !denominator +. weights.(i) *. weights.(i) *. psi.(i) *. psi.(i)
  ) data;
  
  !denominator /. (!numerator *. !numerator)

let grid_search data params_grid k_folds =
  let n = Array.length data in
  let folds = CrossFit.create_folds n k_folds in
  
  List.map (fun params ->
    let scores = Array.map (fun fold ->
      (* Train forest on training data *)
      let train_data = Array.map (fun i -> data.(i)) fold.train_indices in
      let nuisance_estimates = NuisanceEstimator.estimate_conditional_mean train_data in
      let init_theta = 0.0 in
      let forest = RoseRandomForest.create_rose_forest train_data nuisance_estimates init_theta
                    {
                      max_depth = params.max_depth;
                      mtry = params.mtry;
                      min_node_size = params.min_node_size;
                      n_trees = params.n_trees;
                      sample_fraction = 0.632;
                    } in
      
      (* Evaluate on validation data *)
      let test_data = Array.map (fun i -> data.(i)) fold.test_indices in
      let weights = RoseRandomForest.compute_optimal_weights forest test_data in
      let test_nuisance = NuisanceEstimator.estimate_conditional_mean test_data in
      compute_cv_score test_data weights test_nuisance init_theta
    ) folds in
    
    let avg_score = Array.fold_left (+.) 0.0 scores /. float_of_int k_folds in
    { params; score = avg_score }
  ) params_grid