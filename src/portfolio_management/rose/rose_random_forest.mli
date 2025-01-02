type forest_params = {
  max_depth: int;
  mtry: int;
  min_node_size: int;
  n_trees: int;
  sample_fraction: float;
}

val create_rose_forest : Types.observation array -> Types.nuisance_estimates array -> 
                        float -> forest_params -> Types.forest
val compute_optimal_weights : Types.forest -> Types.observation array -> float array
val estimate : Types.observation array -> Types.forest -> float -> Types.estimation_stats