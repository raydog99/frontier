open Torch

(* Core types *)
type markov_state = {
  distribution: Tensor.t;
  covariance: Tensor.t;
  mean: Tensor.t;
}

type markov_chain = {
  states: markov_state array;
  transition_matrix: Tensor.t;
  stationary_dist: Tensor.t;
}

(* Matrix utilities *)
module MatrixUtils : sig
  val norm2 : Tensor.t -> float
  val normalize : Tensor.t -> Tensor.t
  val compute_eigvals : Tensor.t -> Tensor.t * Tensor.t
  val compute_top_eigvec : Tensor.t -> Tensor.t
  val compute_second_eigenval : Tensor.t -> float
end

(* Markov chain *)
module MarkovChain : sig
  val mixing_time : markov_chain -> float -> int
  val is_reversible : markov_chain -> bool
  val sample_trajectory : markov_chain -> int -> int -> int array
end

(* Oja algorithm *)
module Oja : sig
  type config = {
    learning_rate: float;
    decay: float;
    max_iter: int;
  }

  val step_size : int -> config -> float
  val update : Tensor.t -> Tensor.t -> float -> Tensor.t
  val streaming_pca : config -> markov_chain -> Tensor.t -> Tensor.t
end

(* Variance estimation *)
module Variance : sig
  val estimate_variance : markov_chain -> int array -> float
end

(* Mixing time analysis *)
module MixingAnalysis : sig
  val compute_total_variation_distance : Tensor.t -> Tensor.t -> float
  val compute_dmix : markov_chain -> int -> float
  val compute_mixing_time : markov_chain -> float -> int
end

(* Matrix product analysis *)
module MatrixProducts : sig
  val bound_matrix_product : markov_chain -> int -> int -> float -> Tensor.t * Tensor.t
  val analyze_matrix_sequence : markov_chain -> float array -> int -> int -> Tensor.t * Tensor.t
end

(* Covariance analysis *)
module CovarianceAnalysis : sig
  type covariance_bound = {
    cross_covariance: Tensor.t;
    temporal_dependence: float array;
    mixing_effect: float;
    error_estimate: float;
  }

  val analyze_dependent_terms : markov_chain -> int -> covariance_bound
  val analyze_t1_t2_terms : markov_chain -> float -> int -> Tensor.t
end

(* Adaptive mechanisms *)
module AdaptiveMechanisms : sig
  type adaptive_params = {
    step_size: float;
    window_size: int;
    reset_threshold: float;
  }

  val compute_optimal_params : markov_chain -> int -> float -> adaptive_params
  val create_adaptive_algorithm : Oja.config -> markov_chain -> Tensor.t
end

(* Federation components *)
module Federation : sig
  type machine = {
    id: int;
    data_fraction: float;
    local_chain: markov_chain;
    neighbors: int array;
  }

  type network = {
    machines: machine array;
    graph: Tensor.t;
    token_chain: markov_chain;
  }

  val design_transition_matrix : network -> Tensor.t
  val create_token_chain : network -> markov_chain
  val create_federated_algorithm : Oja.config -> network -> Tensor.t
end

(* StreamingPCA *)
module StreamingPCA : sig
  type algorithm_mode = 
    | Standard
    | Adaptive
    | Federated

  val create_algorithm : algorithm_mode -> Oja.config -> markov_chain -> Federation.network option -> Tensor.t
  val compute_error : Tensor.t -> markov_chain -> float
end