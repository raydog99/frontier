let optimize_parameters data nuisance_estimates param_space n_iter =
  let init_configs = List.flatten (
    List.map (fun max_depth ->
      List.map (fun mtry ->
        List.map (fun n_trees ->
          List.map (fun min_node_size ->
            {max_depth; mtry; min_node_size; n_trees}
          ) param_space.min_node_size_range
        ) param_space.n_trees_range
      ) param_space.mtry_range
    ) param_space.max_depth_range
  ) in
  
  let evaluate_config config =
    let forest = RoseRandomForest.create_rose_forest 
      data nuisance_estimates 0.0 config in
    let weights = RoseForest.compute_optimal_weights forest data in
    CrossValidation.compute_cv_score data weights nuisance_estimates 0.0
  in
  
  List.sort (fun (_, s1) (_, s2) -> compare s1 s2) 
    (List.map (fun config -> (config, evaluate_config config)) init_configs)
  |> List.hd |> fst