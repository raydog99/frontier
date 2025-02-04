open Torch

type timestamp = float
type quantile = float

type dataset = {
  y: Tensor.t;       (** Target variable *)
  x: Tensor.t;       (** Covariates matrix *)
  times: timestamp array;
}

type density_estimate = {
  pdf: Tensor.t;
  cdf: Tensor.t;
  support: Tensor.t;
  bandwidth: float;
}

type stationarity_test = {
  is_stationary: bool;
  test_statistic: float;
  critical_values: float array;
  p_value: float;
}

type ergodicity_test = {
  is_ergodic: bool;
  mixing_coefficient: float;
  convergence_rate: float;
}

type model_config = {
  dimension: int;
  num_quantiles: int;
  num_lags: int;
  quantile_levels: float array;
  max_iterations: int;
  convergence_tol: float;
}

type parameters = {
  beta: Tensor.t;     (** Covariate coefficients *)
  theta: Tensor.t;    (** Autoregressive coefficients *)
  lambda: float;      (** Crossing penalty parameter *)
}

type model_state = {
  quantiles: Tensor.t;      (** Current quantile estimates *)
  quantile_history: Tensor.t;  (** Historical quantile values *)
  volatility: Tensor.t;     (** Volatility estimates *)
  inertia: float;          (** Quantile stickiness parameter *)
}

type divergence_components = {
  pinball: float;
  crossing: float;
  volatility: float;
  total: float;
}

module Stationarity : sig
  val compute_rolling_stats : Tensor.t -> int -> Tensor.t array * Tensor.t array
  val compute_autocovariance : Tensor.t -> int -> float array
  val kpss_test : Tensor.t -> stationarity_test
  val adf_test : Tensor.t -> int -> stationarity_test
end

module Ergodicity : sig
  val estimate_mixing_coefficient : Tensor.t -> int -> float
  val test_ergodicity : Tensor.t -> ergodicity_test
end

module Density : sig
  val gaussian_kernel : Tensor.t -> Tensor.t
  val epanechnikov_kernel : Tensor.t -> Tensor.t
  val silverman_bandwidth : Tensor.t -> float
  val boundary_correction : Tensor.t -> float -> float -> (Tensor.t -> Tensor.t) -> Tensor.t
  val estimate_density : ?n_points:int -> Tensor.t -> density_estimate
end

module TensorOps : sig
  val efficient_matmul : Tensor.t -> Tensor.t -> out:Tensor.t option -> Tensor.t
  val update_inplace : Tensor.t -> Tensor.t -> Tensor.t
  val clean_computation : ('a -> 'b) -> 'a -> 'b
end

module Dynamics : sig
  module MultiLag : sig
    val compute_lagged_effects : Tensor.t -> int array -> Tensor.t array
    val aggregate_lag_effects : Tensor.t array -> Tensor.t array -> Tensor.t
  end

  module Volatility : sig
    val estimate_local_volatility : Tensor.t -> int -> Tensor.t
    val compute_volatility_impact : Tensor.t -> Tensor.t -> parameters -> Tensor.t
  end

  val evolve_quantiles : parameters -> model_state -> Tensor.t -> Tensor.t
end

module Constraints : sig
  val compute_crossing_measure : Tensor.t -> Tensor.t -> Tensor.t
  val enforce_ordering : Tensor.t -> float array -> Tensor.t
  val compute_total_crossing : Tensor.t -> Tensor.t
end

module Divergence : sig
  val pinball_divergence : float -> Tensor.t -> Tensor.t -> Tensor.t
  val compute_divergence : parameters -> model_state -> Tensor.t -> model_config -> divergence_components
end

module Adaptation : sig
  val adapt_parameters : parameters -> float array -> parameters
  val update_state : model_state -> Tensor.t -> Tensor.t -> int -> model_state
end

module CMAES : sig
  type strategy_params = private {
    mu: int;
    weights: Tensor.t;
    mueff: float;
    cc: float;
    cs: float;
    c1: float;
    cmu: float;
    damps: float;
  }

  type state = private {
    dimension: int;
    population_size: int;
    params: strategy_params;
    mean: Tensor.t;
    sigma: float;
    pc: Tensor.t;
    ps: Tensor.t;
    C: Tensor.t;
    B: Tensor.t;
    D: Tensor.t;
    eigeneval: int;
    generation: int;
    restarts: int;
    stagnation_count: int;
  }

  module Restart : sig
    type restart_config = {
      max_restarts: int;
      stagnation_tolerance: float;
      improvement_threshold: float;
      increase_popsize: bool;
    }

    val should_restart : state -> restart_config -> float array -> bool
    val reinitialize : state -> increase_popsize:bool -> state
  end

  val create : int -> population_size:int -> state
  
  module Sampling : sig
    val generate_population : state -> Tensor.t * state
    val sample_single : state -> Tensor.t
  end

  module Adaptation : sig
    val update_covariance : state -> population:Tensor.t -> fitness:Tensor.t -> state
  end
end

module Optimization : sig
  type optimization_state = {
    best_solution: Tensor.t;
    best_fitness: float;
    current_state: CMAES.state;
    history: float array;
  }

  val create_optimizer : int -> population_size:int -> optimization_state
  val step : optimization_state -> objective:(Tensor.t -> float) -> optimization_state
  val optimize : objective:(Tensor.t -> float) -> init_state:optimization_state -> 
                config:CMAES.Restart.restart_config -> optimization_state
end