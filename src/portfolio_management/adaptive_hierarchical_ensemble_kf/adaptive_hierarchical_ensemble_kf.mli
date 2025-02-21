open Torch

(* Stable inverse using SVD *)
val stable_inverse : ?rcond:float -> Tensor.t -> Tensor.t

(* Matrix square root via Cholesky *)
val matrix_sqrt : Tensor.t -> Tensor.t

(* Solve linear system with regularization *)
val solve : ?alpha:float -> Tensor.t -> Tensor.t -> Tensor.t

(* State Space Module *)
module StateSpace : sig
  type t = {
    dim: int;
    inner_product_op: Tensor.t option;
  }

  (* Create state space *)
  val create : ?inner_product_op:Tensor.t option -> int -> t

  (* Inner product computation *)
  val inner_product : t -> Tensor.t -> Tensor.t -> float

  (* Norm computation *)
  val norm : t -> Tensor.t -> float

  (* Orthogonalize vector against basis *)
  val orthogonalize : t -> Tensor.t -> Tensor.t array -> Tensor.t

  (* Project vector onto subspace *)
  val project : t -> Tensor.t -> Tensor.t array -> Tensor.t
end

(* Ensemble Module *)
module Ensemble : sig
  type t = {
    members: Tensor.t array;
    space: StateSpace.t;
    weights: float array option;
  }

  (* Create ensemble *)
  val create : ?weights:float array option -> Tensor.t array -> StateSpace.t -> t

  (* Compute weighted ensemble mean *)
  val mean : t -> Tensor.t

  (* Compute ensemble covariance *)
  val covariance : t -> Tensor.t

  (* Compute correlation between ensembles *)
  val correlation : t -> t -> Tensor.t

  (* Split ensemble *)
  val split : t -> t * t

  (* Sample from ensemble *)
  val sample : t -> int -> Tensor.t array

  (* Project ensemble onto basis *)
  val project_to_basis : t -> Tensor.t array -> t
end

(* Proper Orthogonal Decomposition (POD) Module *)
module POD : sig
  type t = {
    basis: Tensor.t;
    singular_values: Tensor.t;
    energy: float array;
    mean: Tensor.t;
  }

  (* Compute POD from snapshot matrix *)
  val compute : ?center:bool -> ?eps:float -> Tensor.t array -> t

  (* Determine number of modes for energy threshold *)
  val number_of_modes : t -> float -> int

  (* Project state onto POD basis *)
  val project : t -> Tensor.t -> int -> Tensor.t

  (* Reconstruct state from coefficients *)
  val reconstruct : t -> Tensor.t -> Tensor.t
end

(* Reduced Basis Module *)
module ReducedBasis : sig
  type t = {
    pod: POD.t;
    n_modes: int;
    space: StateSpace.t;
  }

  (* Create reduced basis *)
  val create : ?energy_threshold:float -> Tensor.t array -> StateSpace.t -> t

  (* Create from existing matrices *)
  val create_from_matrices : Tensor.t -> Tensor.t -> Tensor.t -> t

  (* Project state *)
  val project : t -> Tensor.t -> Tensor.t

  (* Reconstruct state *)
  val reconstruct : t -> Tensor.t -> Tensor.t

  (* Extend basis *)
  val extend : t -> int -> t

  (* Truncate basis *)
  val truncate : t -> int -> t
end

(* Error Estimation Module *)
module ErrorEstimation : sig
  type error_component = {
    local: float;
    global: float;
    propagation: float;
    correlation: float;
  }

  type error_bound = {
    estimate: float;
    upper_bound: float;
    confidence: float;
  }

  type error_history = {
    times: float array;
    errors: error_component array;
    trends: float array;
  }

  (* Compute error components *)
  val analyze_error : 
    high_fidelity:Tensor.t ->
    low_fidelity:Tensor.t ->
    reference:Tensor.t ->
    error_component

  (* Track error history *)
  val track_errors : error_component array -> float -> error_history

  (* Analyze stability *)
  val analyze_stability : error_component array -> bool * float

  (* Error-based Adaptivity Criteria *)
  module AdaptivityCriteria : sig
    type criterion = 
      | ErrorThreshold of float
      | RelativeReduction of float
      | GradientBased of float
      | Combined of criterion list

    (* Check adaptation criterion *)
    val check_criterion : error_bound -> criterion -> bool
  end
end

(* Multi-Level Covariance Module *)
module MultiLevelCovariance : sig
  type level_weights = {
    principal: float;
    control: float;
    ancillary: float;
    cross_correlation: float;
  }

  type covariance_estimate = {
    mean: Tensor.t;
    covariance: Tensor.t;
    cross_terms: Tensor.t array;
    condition_number: float;
    rank: int;
  }

  (* Create level weights *)
  val create_weights : np:int -> nc:int -> na:int -> level_weights

  (* Cross-covariance computation *)
  val cross_covariance : Tensor.t array -> Tensor.t array -> Tensor.t

  (* Multi-level covariance estimation *)
  val estimate : 
    principal:Tensor.t array ->
    control:Tensor.t array ->
    ancillary:Tensor.t array ->
    level_weights ->
    covariance_estimate
end

(* Advanced Regularization Module *)
module AdvancedRegularization : sig
  type regularization_method =
    | Tikhonov of float
    | Spectrum of float
    | LocalizedTapering of float
    | AdaptiveShrinkage of float

  type regularization_stats = {
    original_condition: float;
    regularized_condition: float;
    rank_reduction: int;
    frobenius_change: float;
  }

  (* Apply regularization with statistics *)
  val regularize : 
    Tensor.t -> 
    regularization_method -> 
    Tensor.t * regularization_stats
end

(* Memory-Efficient Adaptation Module *)
module EfficientAdaptation : sig
  type memory_arena = {
    max_total: int;
    max_per_level: int;
    min_required: int;
  }

  type resource_usage = {
    active_memory: int;
    cached_memory: int;
    temporary_memory: int;
  }

  (* Adapt basis efficiently *)
  val adapt_basis : 
    StateSpace.t -> 
    Tensor.t array -> 
    memory_arena -> 
    ReducedBasis.t
end

(* Adaptive Hierarchical Ensemble Kalman Filter *)
module AdaptiveHierarchicalEnKF : sig
  type config = {
    dim: int;
    max_levels: int;
    base_ensemble_size: int;
    initial_basis_size: int;
    error_threshold: float;
    correlation_threshold: float;
    efficiency_threshold: float;
    memory_arena: EfficientAdaptation.memory_arena;
    regularization: AdvancedRegularization.regularization_method;
  }

  type level = {
    id: int;
    basis: ReducedBasis.t;
    ensemble: Ensemble.t;
    parent: level option;
    children: level list;
    error_history: ErrorEstimation.error_history option;
  }

  type t = {
    config: config;
    levels: level array;
    covariance: MultiLevelCovariance.covariance_estimate;
    time: float;
    step: int;
  }

  (* Create initial filter state *)
  val create : config -> Tensor.t -> t

  (* Analysis step *)
  val analysis_step : 
    t -> 
    observation:{data: Tensor.t; noise_cov: Tensor.t} -> 
    t

  (* Prediction step *)
  val prediction_step : 
    t -> 
    (Tensor.t -> float -> Tensor.t) -> 
    float -> 
    t

  (* Complete filter step *)
  val step : 
    t -> 
    observation:{data: Tensor.t; noise_cov: Tensor.t} -> 
    (Tensor.t -> float -> Tensor.t) -> 
    float -> 
    t
end