type params = {
  n_trees: int;
  max_depth: int;
  mtry: int;
  k_folds: int;
  alpha: float;
}

val fit : Types.observation array -> params -> Types.estimation_stats