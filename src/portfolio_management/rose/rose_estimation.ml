type params = {
  n_trees: int;
  max_depth: int;
  mtry: int;
  k_folds: int;
  alpha: float;
}

let fit data params =
  (* Cross-validation for hyperparameter tuning *)
  let params_grid = [
    {max_depth = 5; mtry = 2; min_node_size = 5; n_trees = 100};
    {max_depth = 10; mtry = 3; min_node_size = 10; n_trees = 200};
  ] in
  
  let cv_results = CrossValidation.grid_search data params_grid params.k_folds in
  let best_params = List.fold_left (fun best result ->
    if result.score < best.score then result else best
  ) (List.hd cv_results) cv_results in
  
  (* Estimate nuisance functions *)
  let nuisance_estimates = NuisanceEstimator.estimate_all data Models.PartiallyLinear in
  
  (* Create and train forest with best parameters *)
  let forest = RoseRandomForest.create_rose_forest data nuisance_estimates 0.0 
                {
                  max_depth = best_params.params.max_depth;
                  mtry = best_params.params.mtry;
                  min_node_size = best_params.params.min_node_size;
                  n_trees = best_params.params.n_trees;
                  sample_fraction = 0.632;
                } in
  
  (* Final estimation *)
  RoseForest.estimate data forest params.alpha