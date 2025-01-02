type parameter_space = {
  max_depth_range: int list;
  mtry_range: int list;
  n_trees_range: int list;
  min_node_size_range: int list;
}

val optimize_parameters : Types.observation array -> Types.nuisance_estimates array -> 
                        parameter_space -> int -> RoseRandomForest.forest_params