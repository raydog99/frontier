open Torch

(** Core matrix operations *)
module Matrix : sig
  type t
  val create : int -> int -> t
  val dims : t -> int * int
  val get : t -> int -> int -> float
  val set : t -> int -> int -> float -> t
  val matmul : t -> t -> t
  val transpose : t -> t
  val is_symmetric : t -> bool
  val is_skew_symmetric : t -> bool
  val of_lists : float list list -> t
  val to_float_array2 : t -> float array array
end

(** Handling observed matrices with missing entries *)
module ObservedMatrix : sig
  type t = {
    data: Matrix.t;
    mask: Matrix.t;
  }
  
  val create : Matrix.t -> Matrix.t -> t
  val dims : t -> int * int
  val observed_proportion : t -> float
  val get : t -> int -> int -> float option
  val to_dense : t -> Matrix.t
  val get_mask : t -> Matrix.t
  val create_random_mask : int -> int -> float -> Matrix.t
end

(** SVD and numerical stability *)
module NumericalStability : sig
  type svd_result = {
    u: Matrix.t;
    s: Matrix.t;
    vt: Matrix.t;
  }

  val stable_svd : Matrix.t -> epsilon:float -> Matrix.t * Matrix.t * Matrix.t
  val condition_number : Matrix.t -> float
  val stable_matmul : Matrix.t -> Matrix.t -> epsilon:float -> Matrix.t
end

(** Basic statistics *)
module Stats : sig
  val mse : Matrix.t -> Matrix.t -> float
  val nuclear_norm : Matrix.t -> float
  val frobenius_norm : Matrix.t -> float
  val mean : Matrix.t -> float
  val variance : Matrix.t -> float
end

(** Interval scaling and bounds handling *)
module IntervalScaling : sig
  type interval = {
    lower: float;
    upper: float;
  }

  type scaling_params = {
    original_interval: interval;
    scaled_interval: interval;
    scale_factor: float;
    offset: float;
  }

  val compute_scaling_params : original_interval:interval -> target_interval:interval -> scaling_params
  val scale_matrix : Matrix.t -> scaling_params -> Matrix.t
  val inverse_scale_matrix : Matrix.t -> scaling_params -> Matrix.t
  val clip_values : Matrix.t -> interval -> Matrix.t
end

(** Variance handling and analysis *)
module VarianceHandling : sig
  type variance_info = {
    known_variance: bool;
    sigma_sq: float option;
    estimated_variance: float;
    q_factor: float;
    confidence_level: float;
  }

  val estimate_variance : ObservedMatrix.t -> float
  val compute_q_factor : float -> float -> float
  val create_variance_info : ?sigma_sq:float -> ObservedMatrix.t -> variance_info
end

(** Core USVT algorithm *)
module USVT : sig
  type config = {
    eta: float;
    bound: float;
    use_variance: bool;
    adaptive_threshold: bool;
  }

  type t = {
    config: config;
    estimate: Matrix.t;
  }

  val default_config : config
  val estimate : ?config:config -> ObservedMatrix.t -> t
end

(** Enhanced USVT with improved variance handling *)
module EnhancedUSVT : sig
  type config = {
    base_config: USVT.config;
    min_rank: int option;
    max_rank: int option;
    convergence_threshold: float;
    use_iterative_refinement: bool;
  }

  val default_config : config
  val estimate : ?config:config -> ObservedMatrix.t -> USVT.t
end

(** Stochastic Block Model *)
module StochasticBlockModel : sig
  type config = {
    n: int;
    k: int;
    sparsity: float;
    growing_blocks: bool;
    min_block_size: int;
  }

  type block_structure = {
    assignments: int array;
    sizes: int array;
    prob_matrix: Matrix.t;
  }

  val create_stochastic_model : config -> block_structure
  val estimate_blocks : ObservedMatrix.t -> config -> block_structure
end

(** Distance Matrix *)
module DistanceMatrix : sig
  module MetricSpace : sig
    module type METRIC = sig
      type t
      val distance : t -> t -> float
      val ball_covering : t list -> float -> t list list
      val diameter : t list -> float
      val is_compact : t list -> bool
    end

    module EuclideanMetric : METRIC with type t = float array
  end

  type point = float array
  
  type t = {
    points: point array;
    metric: (module MetricSpace.METRIC with type t = point);
  }

  val create : point array -> (module MetricSpace.METRIC with type t = point) -> t
  val compute_distance_matrix : t -> Matrix.t
  val estimate : ObservedMatrix.t -> USVT.t
end

(** Graphon estimation *)
module Graphon : sig
  type t = {
    f: float -> float -> float;
    is_symmetric: bool;
    is_measurable: bool;
    support: float * float;
  }

  type discrete_graphon = {
    matrix: Matrix.t;
    n: int;
  }

  val step_graphon : Matrix.t -> t
  val estimate : ObservedMatrix.t -> USVT.t
end

(** Bradley-Terry model *)
module BradleyTerry : sig
  type player = {
    id: int;
    name: string option;
    initial_rating: float option;
  }

  type match_result = {
    player1: int;
    player2: int;
    outcome: float;
    weight: float;
    timestamp: float option;
  }

  type model = {
    n_players: int;
    players: player array;
    strength_matrix: Matrix.t;
    ratings: float array;
    uncertainty: float array;
  }

  type config = {
    allow_draws: bool;
    temporal_weighting: bool;
    min_matches: int;
    regularization: float;
    convergence_tol: float;
    max_iterations: int;
  }

  val default_config : config
  val create_player : ?name:string -> ?initial_rating:float -> int -> player
  val create_model : player array -> model
  val estimate : ?config:config -> model -> match_result array -> model
end