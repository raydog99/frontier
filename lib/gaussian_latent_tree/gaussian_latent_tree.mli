open Torch

module Tree : sig
  type node = 
    | Leaf of int  
    | Internal of int

  type edge = {
    source: node;
    target: node;
    correlation: float Tensor.t;
  }

  type t = {
    nodes: node list;
    edges: edge list;
    n_leaves: int;
    n_internal: int;
  }

  val create : node list -> edge list -> t
  val get_neighbors : node -> t -> node list
  val get_edge : node -> node -> t -> edge option
  val check_degree_constraints : t -> bool
  val normalize_tree : t -> t
  val find_path : node -> node -> t -> node list option
  val partition_leaves : node -> t -> int list
  val compute_path_product : EM.params -> int -> t -> float
  val get_subtree : node -> t -> t
  val is_connected : t -> bool
  val get_path_length : node -> node -> t -> int option
end

val inverse : float Tensor.t -> float Tensor.t
val compute_jacobian : float Tensor.t -> float Tensor.t -> float Tensor.t
val condition_gaussian : float Tensor.t -> float Tensor.t -> int list -> float Tensor.t
val compute_conditional_mean : EM.params -> float Tensor.t -> int -> int -> float Tensor.t
val compute_conditional_variance : EM.params -> int -> Tree.t -> float Tensor.t
val compute_conditional_expectation : EM.params -> int -> int -> Tree.t -> float
val compute_conditional_expectation_squared : EM.params -> int -> Tree.t -> float
val compute_information_parameter : EM.params -> int -> Tree.t -> float
val compute_leaf_linear_combination : EM.params -> int list -> Tree.t -> float
val compute_conditional_covariance : EM.params -> int list -> Tree.t -> float Tensor.t
val compute_precision_matrix : EM.params -> Tree.t -> float Tensor.t
val compute_marginal_distribution : EM.params -> int list -> Tree.t -> float Tensor.t * float Tensor.t

module EM : sig
  type params = {
    correlations: float Tensor.t;
    variances: float Tensor.t;
  }

  val create_params : int -> params
  val e_step : params -> float Tensor.t -> Tree.t -> float Tensor.t * float Tensor.t
  val m_step : float Tensor.t -> (float Tensor.t * float Tensor.t) -> Tree.t -> params
  val fit : ?max_iter:int -> ?tol:float -> params -> float Tensor.t -> Tree.t -> params
  val update_parameters : params -> Tree.t -> params
  val fit : ?max_iter:int -> ?tol:float -> params -> Tree.t -> params
  val update_parameters : params -> Tree.t -> float Tensor.t -> params
  val is_trivial_point : params -> Tree.t -> bool
  val fit_with_guarantees : 
    ?max_iter:int -> ?tol:float -> ?alpha:float -> ?beta:float -> 
    params -> float Tensor.t -> Tree.t -> params
end

module ConvergenceAnalysis : sig
  type convergence_state = {
    correlations: float Tensor.t;
    iteration: int;
    log_likelihood: float;
    stationary: bool;
  }

  val check_stationarity : EM.params -> EM.params -> float -> bool
  val characterize_stationary_points : EM.params -> Tree.t -> bool * bool array
  val verify_convergence_bounds : EM.params -> bool
  val verify_no_interior_stationary : EM.params -> Tree.t -> EM.params -> bool
  val analyze_convergence : 
    ?max_iter:int -> ?tol:float -> EM.params -> EM.params -> float Tensor.t -> Tree.t -> 
    (EM.params, string) result
end

module TaylorSeriesAnalysis : sig
  type expansion_terms = {
    constant: float;
    first_order: float Tensor.t;
    second_order: float Tensor.t;
  }

  val compute_derivatives : EM.params -> Tree.t -> int -> expansion_terms
  val analyze_near_zero : EM.params -> Tree.t -> bool
  val analyze_near_special_point : EM.params -> Tree.t -> int -> bool
  val analyze_convergence : EM.params -> Tree.t -> (EM.params, string) result
  val verify_taylor_accuracy : EM.params -> Tree.t -> bool
end

module GeneralTreeModel : sig
  type node_params = {
    mean: float Tensor.t;
    variance: float Tensor.t;
  }

  type model = {
    tree: Tree.t;
    node_params: node_params array;
    edge_correlations: float Tensor.t;
  }

  val create : Tree.t -> model
  val compute_likelihood : model -> float Tensor.t -> float
  val em_step : model -> float Tensor.t -> model
  val fit : ?max_iter:int -> ?tol:float -> model -> float Tensor.t -> model
end

module GeneralTreeAnalysis : sig
  type sufficient_statistics = {
    first_moments: float Tensor.t;
    second_moments: float Tensor.t;
    empirical_cov: float Tensor.t;
  }

  val verify_conditional_conservation : EM.params -> Tree.t -> bool
  val verify_likelihood_stationarity : EM.params -> Tree.t -> float Tensor.t -> bool
  val verify_conditional_expectations : EM.params -> EM.params -> Tree.t -> bool
  val verify_linear_combination_property : EM.params -> Tree.t -> bool
  val verify_invertibility_conditions : EM.params -> Tree.t -> bool
  val verify_convergence : EM.params -> EM.params -> Tree.t -> bool
end

module FiniteSampleAnalysis : sig
  type complexity_bounds = {
    sample_complexity: int;
    iteration_complexity: int;
  }

  val compute_complexity_bounds : int -> float -> float -> float -> float -> complexity_bounds
  val verify_guarantees : EM.params -> EM.params -> float Tensor.t -> int -> float -> bool
end