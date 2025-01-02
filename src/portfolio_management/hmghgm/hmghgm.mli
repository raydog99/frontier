open Torch

val mahalanobis_distance : Tensor.t -> Tensor.t -> Tensor.t -> float
val l1_offdiag : Tensor.t -> float
val normalize_covariance : Tensor.t -> Tensor.t
val constrain_covariance : Tensor.t -> Tensor.t
val log_sum_exp : float array -> float
val safe_inverse : Tensor.t -> Tensor.t
val stable_covariance_update : Tensor.t -> float array -> Tensor.t

module GH : sig
  type t = private {
    mu: Tensor.t;
    sigma: Tensor.t;
    theta: Tensor.t;
    lambda: float;
    chi: float;
    psi: float;
    d: int;
  }

  val create : mu:Tensor.t -> sigma:Tensor.t -> lambda:float -> 
               chi:float -> psi:float -> t
  val density : t -> Tensor.t -> float
  val mixture_representation : t -> Tensor.t
  val bessel_k : float -> float -> float

  module Hierarchical : sig
    type state = {
      w: float;
      y: Tensor.t
    }
    val sample : t -> state
  end
end

module GIG : sig
  type t = {
    lambda: float;
    chi: float;
    psi: float;
  }

  val density : t -> float -> float
  val sample : lambda:float -> chi:float -> psi:float -> float
  val expected_values : t -> Tensor.t -> Tensor.t -> Tensor.t -> int -> 
                       float * float * float
end

module HMM : sig
  type t = private {
    k: int;
    d: int;
    pi: Tensor.t;
    transition: Tensor.t;
    gh_params: GH.t array;
    theta: Tensor.t array;
  }

  val create : k:int -> d:int -> pi:Tensor.t -> transition:Tensor.t -> 
               gh_params:GH.t array -> t
  val emission_probability : t -> Tensor.t -> int -> float
  val sequence_probability : t -> Tensor.t -> float
  val sample : t -> int -> Tensor.t * int array

  module StateGraph : sig
    type edge = {
      from_node: int;
      to_node: int;
      weight: float;
    }

    type t = {
      state: int;
      nodes: int array;
      edges: edge list;
      precision: Tensor.t;
    }

    val from_precision : int -> Tensor.t -> t
    val is_conditionally_independent : t -> int -> int -> int list -> bool
  end

  val update_state_parameters : t -> Tensor.t -> float array -> int -> 
                               Tensor.t * StateGraph.t * GH.t
end

module ForwardBackward : sig
  type scaled_probs = {
    alpha: Tensor.t;
    beta: Tensor.t;
    scaling: float array;
  }

  val forward : HMM.t -> Tensor.t -> Tensor.t * float array * float array
  val backward : HMM.t -> Tensor.t -> float array -> float array -> Tensor.t
  val smooth : HMM.t -> Tensor.t -> scaled_probs * Tensor.t * Tensor.t
end

module ECME : sig
  type params = {
    hmm: HMM.t;
    rho: float;
    state_weights: float array;
  }

  val e_step : params -> Tensor.t -> Tensor.t * Tensor.t
  val cm_step1 : params -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
  val cm_step2 : params -> Tensor.t -> Tensor.t -> GH.t array
  val cm_step3 : params -> Tensor.t -> Tensor.t -> GH.t array
  val optimize : init_params:params -> data:Tensor.t -> 
                max_iter:int -> tol:float -> params
end

module PenalizedECME : sig
  type params = {
    hmm: HMM.t;
    rho: float;
    state_weights: float array;
    max_iter: int;
    tol: float;
  }

  val penalized_likelihood : params -> Tensor.t -> float array -> int -> float
  val weighted_covariance : Tensor.t -> float array -> Tensor.t -> 
                           float array -> Tensor.t
  val graphical_lasso_penalized : Tensor.t -> float -> float -> int -> Tensor.t
  val optimize_penalized : init_params:params -> data:Tensor.t -> 
                          max_iter:int -> tol:float -> params
  val select_penalty_parameter : Tensor.t -> float array -> params
end