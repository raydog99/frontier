val compute_split_criterion : Types.observation array -> Types.nuisance_estimates array -> float -> int array -> int -> float -> float
val find_best_split : Types.observation array -> Types.nuisance_estimates array -> float -> Types.node -> int -> int * float
val partition_data : int array -> int -> float -> Types.observation array -> int array * int array