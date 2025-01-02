val create_leaf : int array -> (float * float) array -> Types.split_stats -> Types.node
val build_tree : Types.observation array -> Types.nuisance_estimates array -> float -> 
                int array -> (float * float) array -> Types.split_stats -> 
                int -> int -> int -> Types.node
val compute_node_weights : Types.node -> float array -> float
val check_split_validity : int -> int -> int -> float -> bool