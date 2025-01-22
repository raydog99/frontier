open Torch

(** Core types *)
type node_id = int
type weight = float

type edge = {
  source: node_id;
  target: node_id;
  weight: weight;
}

type network = {
  nodes: node_id array;
  adj_matrix: Tensor.t;
  weights_matrix: Tensor.t;
  properties: network_properties;
}

and network_properties = {
  in_degree_distribution: int array;
  out_degree_distribution: int array;
  weight_distribution: int array;
  clustering_coefficients: Tensor.t;
}

type assortativity_measures = {
  in_in: float;
  in_out: float;
  out_in: float;
  out_out: float;
  total: float;
}

(** Markov Property *)
module MarkovProperty : sig
  type state_history = {
    current: Tensor.t;
    past: Tensor.t list;
    max_order: int;
  }

  val create_history : max_order:int -> n_series:int -> state_history
  val update_history : state_history -> Tensor.t -> state_history
  val check_markov_property : state_history -> Tensor.t -> Tensor.t -> Tensor.t
end

(** State Space *)
module StateSpace : sig
  type state = int
  type state_space = {
    n_states: int;
    n_series: int;
  }

  val create : n_states:int -> n_series:int -> state_space
  val discretize_returns : Tensor.t -> int -> Tensor.t
end

(** Joint Distribution *)
module JointDistribution : sig
  type distribution_params = {
    eps: float;
    min_prob: float;
    regularization: float;
    check_validity: bool;
  }

  val compute_joint : Tensor.t -> distribution_params -> Tensor.t
  val compute_conditional : Tensor.t -> distribution_params -> Tensor.t
end

(** MTD Estimation *)
module MTDEstimation : sig
  type estimation_params = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    lambda_constraint_weight: float;
  }

  val estimate : estimation_params -> Tensor.t -> Tensor.t array array * Tensor.t
end

(** Network Construction *)
module NetworkConstruction : sig
  type network_params = {
    min_weight: float;
    remove_self_loops: bool;
    weight_normalization: [`None | `Row | `Column | `Global];
  }

  val create_from_mtd : 
    < lambda_weights : Tensor.t; .. > -> 
    network_params -> 
    network
end

(** Global Assortativity *)
module GlobalAssortativity : sig
  type correlation_params = {
    normalization: [`Standard | `Weighted];
    omega_handling: [`Global | `Local];
    excess_type: [`Simple | `Weighted];
  }

  val compute : network -> correlation_params -> assortativity_measures
end

(** Local Assortativity *)
module LocalAssortativity : sig
  module LocalAssortExtension : sig
    val compute_local_contribution : 
      network -> 
      int -> 
      GlobalAssortativity.correlation_params -> 
      float

    val compute_all : 
      network -> 
      GlobalAssortativity.correlation_params -> 
      Tensor.t
  end
end

(** Enhanced Sabek *)
module EnhancedSabek : sig
  type edge_params = {
    normalization: [`Local | `Global];
    weight_handling: [`Raw | `Normalized];
    neighbor_type: [`Direct | `Extended];
  }

  val default_params : edge_params
  val compute_node_assortativity : network -> int -> edge_params -> float
end

(** Enhanced Peel *)
module EnhancedPeel : sig
  type multiscale_params = {
    n_scales: int;
    alpha_range: float * float;
    integration_method: [`Uniform | `Weighted];
    scale_weights: float array option;
  }

  val compute_local : network -> multiscale_params -> Tensor.t
end

(** Portfolio *)
module Portfolio : sig
  type portfolio_weights = Tensor.t
  
  type optimization_result = {
    weights: portfolio_weights;
    objective_value: float;
    network_metrics: network_metrics;
    convergence: convergence_info;
  }
  and network_metrics = {
    assortativity: float;
    centrality: float;
    risk_contribution: float;
  }
  and convergence_info = {
    iterations: int;
    final_grad_norm: float;
    converged: bool;
  }

  module Utils : sig
    val portfolio_return : Tensor.t -> Tensor.t -> float
    val portfolio_variance : Tensor.t -> Tensor.t -> float
    val portfolio_sharpe : Tensor.t -> Tensor.t -> Tensor.t -> float -> float
  end

  module Constraints : sig
    type constraints = {
      min_weight: float;
      max_weight: float;
      sum_weights: float;
      long_only: bool;
    }

    val default : constraints
    val apply_constraints : constraints -> Tensor.t -> Tensor.t
  end
end

(** Portfolio Optimizer *)
module PortfolioOptimizer : sig
  type optimization_params = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    risk_aversion: float;
    risk_free_rate: float;
    network_weight: float;
  }

  val optimize : 
    optimization_params -> 
    Tensor.t -> 
    Tensor.t -> 
    network -> 
    Tensor.t option -> 
    Portfolio.optimization_result
end

(** Network Portfolio Integration *)
module NetworkPortfolioIntegration : sig
  type optimization_config = {
    network_params: < n_scales : int; .. >;
    portfolio_params: PortfolioOptimizer.optimization_params;
    estimation_params: MTDEstimation.estimation_params;
  }

  type optimization_result = {
    portfolio: Portfolio.optimization_result;
    network_analysis: network_analysis;
    validation: validation_info;
  }
  type network_analysis = {
    global_assortativity: assortativity_measures;
    local_assortativity: Tensor.t;
    multiscale_measures: Tensor.t;
  }
  type validation_info = {
    warnings: string list;
    diagnostics: diagnostic_info;
  }
  type diagnostic_info = {
    numerical_stability: bool;
    constraint_violation: bool;
    convergence_quality: float;
  }

  val optimize : 
    optimization_config -> 
    Tensor.t -> 
    Tensor.t -> 
    network -> 
    optimization_result
end