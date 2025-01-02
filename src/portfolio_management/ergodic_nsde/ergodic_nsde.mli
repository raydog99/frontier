open Torch

type node_id = int
type edge = node_id * node_id

type graph = {
  nodes: node_id list;
  edges: edge list;
  adjacency: (node_id * node_id) list
}

type parameters = {
  alpha: float Tensor.t;
  beta: float Tensor.t;
  weights: float Tensor.t;
}

type state = {
  x: float Tensor.t;
  time: float;
}

type estimation_options = {
  delta: float;
  threshold: float;
  learning_rate: float;
  max_iter: int;
  tolerance: float;
  sparsity_threshold: float;
  n_samples: int;
  regularization: float;
  dt: float;
}

val diag : float Tensor.t -> float Tensor.t
val adjacency_matrix : graph -> float Tensor.t
val normalize_adjacency : float Tensor.t -> float Tensor.t
val laplacian : graph -> float Tensor.t

val drift : state -> parameters -> graph -> float Tensor.t
val diffusion : state -> parameters -> float Tensor.t
val step : state -> parameters -> graph -> float -> state
val simulate : state -> parameters -> graph -> int -> float -> state list

(** Quasi-likelihood estimation *)
val logdet_stable : float Tensor.t -> float Tensor.t
val score : state list -> parameters -> graph -> float -> parameters
val quasi_likelihood : state list -> parameters -> graph -> float -> float

(** Graph structure estimation *)
val estimate_adjacency : float Tensor.t -> float -> float Tensor.t
val reconstruct_graph : parameters -> float -> graph
val is_subgraph : graph -> graph -> bool

(** Stability analysis *)
val check_lyapunov_stability : parameters -> graph -> bool
val verify_ergodicity : state list -> parameters -> graph -> bool

(** Parameter estimation *)
val block_optimize : state list -> graph -> parameters -> parameters
val optimize_params : state list -> graph -> parameters -> float -> int -> float -> parameters
val estimate : state list -> graph -> parameters

(** Main interface *)
val estimate_network_sde : ?known_graph:bool -> state list -> graph -> estimation_options -> (parameters, string) result
val verify_system : state list -> parameters -> graph -> bool

module AdaptiveLasso : sig
  type lasso_params = {
    lambda: float;
    delta: float;
    threshold: float;
    learning_rate: float;
    max_iter: int;
    tolerance: float;
  }

  val compute_adaptive_weights : parameters -> float -> float Tensor.t
  val proximal_update : parameters -> parameters -> float Tensor.t -> float -> float -> parameters
  val estimate : state list -> graph -> lasso_params -> parameters
end