open Torch

type split_stats = {
  left_treatment_prop: float;
  right_treatment_prop: float;
  left_size: int;
  right_size: int;
  left_kernel_matrix: Tensor.t;
  right_kernel_matrix: Tensor.t
}

type tree_params = {
  max_depth: int;
  min_samples_leaf: int;
  max_features: int;
  honesty_fraction: float;
  regularization: float;
}

val build_tree : sample -> tree_params -> tree_node
val build_forest : sample -> int -> int -> forest
val predict : forest -> Tensor.t -> Tensor.t
val get_weights : forest -> Tensor.t -> int -> float array
val find_best_split : sample -> int array -> tree_params -> (int * float * int array * int array) option
val calculate_node_stats : sample -> int array -> split_stats
val evaluate_split : sample -> int array -> int array -> float
val get_leaf_node : tree_node -> Tensor.t -> tree_node
val random_feature_subset : int -> int -> int array