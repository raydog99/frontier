open Torch

type sample = {
  features : Tensor.t;
  treatment : Tensor.t;
  outcome : Tensor.t
}

type tree_node = {
  split_feature : int option;
  split_value : float option;
  left_child : tree_node option;
  right_child : tree_node option;
  samples : int array;
}

type forest = {
  trees : tree_node array;
  n_trees : int;
  n_groups : int;
}