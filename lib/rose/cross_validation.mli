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

val compute_cv_score : Types.observation array -> float array -> 
                      Types.nuisance_estimates array -> float -> float
val grid_search : Types.observation array -> cv_params list -> int -> cv_result list