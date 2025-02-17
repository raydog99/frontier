open Torch

type optimization_params = {
  max_iter: int;
  tolerance: float;
  learning_rate: float;
  momentum: float;
}

type tensor_pair = {
  first: Tensor.t;
  second: Tensor.t;
}

module AssetHoldings : sig
  type state = {
    value: float;
    weights: Tensor.t;
    returns: Tensor.t;
    costs: Tensor.t;
  }

  val compute_next_value : 
    float -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float

  val update_state : 
    state -> Tensor.t -> state

  val compute_stability : 
    state -> Tensor.t -> float -> float -> float
end

module TradingConstraints : sig
  type t = {
    leverage_limit: float;
    adjustment_limit: float;
    holding_limits: Tensor.t;
    stability_threshold: float;
  }

  val check_leverage : Tensor.t -> float -> bool

  val check_holdings : Tensor.t -> Tensor.t -> bool

  val check_stability : 
    Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> bool

  val check_constraints : 
    t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> bool

  val project : 
    t -> Tensor.t -> Tensor.t -> Tensor.t
end

module DistributionalRobust : sig
  type confidence_set = {
    a0: Tensor.t;
    d0: Tensor.t;
    a1: Tensor.t;
    d1: Tensor.t;
    support: tensor_pair;
  }

  val create_moment_based : 
    Tensor.t -> float -> confidence_set

  val compute_objective : 
    (Tensor.t -> Tensor.t -> float) -> 
    Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> float

  val solve_minimum_case : 
    confidence_set -> Tensor.t -> Tensor.t
end

module SeparableUtility : sig
  type t = {
    phi1: Tensor.t -> Tensor.t;
    phi2: Tensor.t -> Tensor.t;
    phi1_derivative: Tensor.t -> Tensor.t;
    phi2_derivative: Tensor.t -> Tensor.t;
    phi1_second_derivative: Tensor.t -> Tensor.t;
    phi2_second_derivative: Tensor.t -> Tensor.t;
    alpha: float;
    beta: float;
  }

  val create_log : 
    alpha:float -> beta:float -> t

  val create_power : 
    alpha:float -> beta:float -> gamma:float -> t

  val evaluate : 
    t -> Tensor.t -> Tensor.t -> float
end

module Hyperplane : sig
  type t = {
    a: float;
    b: float;
    gamma: float;
    x_point: float;
    c_point: float;
  }

  val create : 
    (Tensor.t -> Tensor.t -> float) -> float -> float -> t

  val evaluate : 
    t -> float -> float -> float
end

module Reliability : sig
  type reliability_components = {
    reliability_x: float;
    reliability_c: float;
    total_reliability: float;
  }

  val analyze_reliability_monotonicity : 
    SeparableUtility.t -> 
    float -> float -> float -> float -> 
    bool * bool

  val compute_max_reliability : 
    SeparableUtility.t -> 
    Hyperplane.t -> 
    float array -> float array -> int -> 
    reliability_components
end

module PartitionRefinement : sig
  type partition_point = {
    x: float;
    c: float;
    reliability: float;
  }

  val compute_successive_points : 
    SeparableUtility.t -> partition_point -> float -> partition_point

  val compute_log_utility_points : 
    current_x:float -> 
    current_c:float -> 
    reliability_x:float -> 
    reliability_c:float -> 
    partition_point
end

module RobustPortfolio : sig
  type t = {
    utility: SeparableUtility.t;
    hyperplanes: Hyperplane.t array;
    partitions: PartitionRefinement.partition_point array;
    constraints: TradingConstraints.t;
    confidence: DistributionalRobust.confidence_set;
  }

  val create : 
    returns:Tensor.t ->
    confidence_level:float ->
    constraints:TradingConstraints.t ->
    utility_type:[`Log | `Power of float] ->
    alpha:float ->
    beta:float ->
    reliability_tol:float ->
    t

  val optimize : 
    t -> Tensor.t -> Tensor.t -> Tensor.t
end