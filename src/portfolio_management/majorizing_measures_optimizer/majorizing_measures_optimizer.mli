open Torch

(* Core types *)
type metric_space = {
  points: tensor;
  distances: tensor;
  n_points: int;
}

type measure = {
  weights: tensor;
  support: tensor;
}

type chaining_functional = {
  f: float -> float;
  f_inv: float -> float;
  is_log_concave: bool;
}

type ball = {
  center: int;
  radius: float;
  metric: metric_space;
}

type 'a tree_node = {
  data: 'a;
  children: 'a tree_node list;
}

type complete_packing_node = {
  vertices: tensor;
  chi: int;
  center: int;
  radius: float;
  diameter: float;
  separation: float;
  parent_radius: float;
  children: complete_packing_node list;
  weight: float;
}

type complete_node = {
  vertices: tensor;
  m_label: int;
  sigma: int;
  center: int;
  diameter: float;
  children: complete_node list;
}

(* Core modules *)
module Measure : sig
  val create : tensor -> measure
  val measure_subset : measure -> tensor -> tensor
  val measure_ball : measure -> ball -> tensor
  val is_normalized : measure -> bool
  val normalize : measure -> measure
end

module Ball : sig
  val create : metric_space -> int -> float -> ball
  val contains : ball -> int -> bool
  val get_points : ball -> tensor
  val diameter : metric_space -> tensor -> tensor
end

module Metric : sig
  val create : tensor -> metric_space
  val distance : metric_space -> int -> int -> tensor
  val closest_point : metric_space -> int -> tensor -> tensor
end

module ChainingFunctional : sig
  val create_gaussian : unit -> chaining_functional
  val evaluate : chaining_functional -> float -> float
  val check_submultiplicative : chaining_functional -> float -> float -> bool
  val check_bounds : chaining_functional -> float -> bool
end

(* High-level interface *)
module Interface : sig
  val compute_packing_tree : tensor -> tensor -> (complete_packing_node * float, string) result
  
  type tree_stats = {
    size: int;
    depth: int;
    branching_factor: float;
    diameter: float;
    separation: float;
  }
  
  val analyze_tree : complete_packing_node -> metric_space -> (tree_stats, string) result
  val create_gaussian_tree : tensor -> tensor -> (float, string) result
  val optimize_existing_tree : complete_packing_node -> metric_space -> measure -> complete_packing_node
end

(* Analysis utilities *)
module Analysis : sig
  type tree_stats = {
    size: int;
    depth: int;
    branching_factor: float;
    diameter: float;
    separation: float;
  }

  val compute_properties : complete_packing_node -> tree_stats
  val analyze_performance : (unit -> 'a) -> int -> {
    input_size: int;
    time_taken: float;
    result: 'a;
  }
end