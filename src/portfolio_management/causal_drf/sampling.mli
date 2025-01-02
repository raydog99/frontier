open Torch

type half_sample = {
  indices: int array;
  tree_assignments: (int * int array) array;
  weights: float array;
  treatment_balance: float;
}

val generate_half_samples : sample -> int -> int -> half_sample array
val estimate_uncertainty : half_sample array -> Tensor.t array -> Tensor.t -> float -> 
  Tensor.t * float
val generate_balanced_sample : sample -> half_sample
val calculate_group_variance : sample -> half_sample array -> Tensor.t array -> Tensor.t