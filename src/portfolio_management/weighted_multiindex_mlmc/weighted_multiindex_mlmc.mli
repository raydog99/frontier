open Torch

module type SDE = sig
  type model = GBM | IGBM | CIR

  type t = {
    model: model;
    r: float;
    init_price: float;
    time_horizon: float;
  }

  val drift : model -> float -> float
  val volatility : model -> float -> float
  val volatility_derivative : model -> float -> float
end

module type Discretization = sig
  type scheme = Euler | Milstein

  val simulate_path : 
    SDE.t -> scheme -> int -> float -> Tensor.t -> Tensor.t
end

module type Estimator = sig
  type t = {
    sde: SDE.t;
    payoff: Payoff.t;
    scheme: Discretization.scheme;
    level: int;
    base: int;
  }

  type level_data = {
    estimator: t;
    variance: float;
    correlation: float;
    cost: float;
    samples: int;
  }

  val create : SDE.t -> Payoff.t -> Discretization.scheme -> int -> int -> t
  val steps : t -> int
  val dt : t -> float
  val estimate : t -> int -> float
  val single_path : t -> float
end

module type MLMC = sig
  type t = {
    levels: Estimator.level_data array;
    target_variance: float;
  }

  val create : SDE.t -> Discretization.scheme -> Payoff.t -> float -> t
  val compute_optimal_samples : t -> int array
  val estimate : t -> float
end

module type WeightedMLMC = sig
  type weight_params = {
    theta: float array;
    alpha: float array;
    beta: float array;
    delta: float array;
    effort: float array;
  }

  type t = {
    levels: Estimator.level_data array;
    weights: weight_params;
    target_variance: float;
  }

  val create : SDE.t -> Discretization.scheme -> Payoff.t -> float -> t
  val compute_optimal_weights : t -> weight_params
  val estimate : t -> float
end

module type MultiIndex = sig
  type t = int array
  
  val zero : int -> t
  val ( <= ) : t -> t -> bool
  val ( < ) : t -> t -> bool
  val box_minus : t -> t list
  val box_plus : t -> t list
  val min_entry : t -> int
end

module type MultiIndexMLMC = sig
  type t = {
    dim: int;
    estimators: Estimator.level_data array;
    target_variance: float;
  }

  val create : int -> SDE.t -> Discretization.scheme -> Payoff.t -> float -> t
  val compute_optimal_samples : t -> int array
  val estimate : t -> float
end

module type WeightedMultiIndexMLMC = sig
  type multi_weight = {
    theta: float array array;
    alpha: float;
    beta: float;
    delta: float;
  }

  type t = {
    dim: int;
    estimators: Estimator.level_data array;
    weights: multi_weight array;
    target_variance: float;
  }

  val create : int -> SDE.t -> Discretization.scheme -> Payoff.t -> float -> t
  val compute_optimal_weights : t -> multi_weight array
  val estimate : t -> float
end

module type Optimization = sig
  type objective = {
    f: float array -> float;
    grad: float array -> float array;
  }

  val minimize : ?max_iter:int -> ?tol:float -> objective -> float array -> float array
  val trust_region_optimize : objective -> float array -> float array
  val coordinate_descent : objective -> float array -> float array
end

module type PathGeneration = sig
  type path_features = {
    mean_reversion: float option;
    jumps: (float * float) list;
    seasonality: float -> float;
    volatility_clustering: bool;
  }

  val generate_path : 
    SDE.t -> path_features -> int -> float -> Tensor.t -> Tensor.t
end

module type NumericalSchemes = sig
  type step_config = {
    order: int;
    stability_factor: float;
    error_tolerance: float;
    max_steps: int;
  }

  module MultiStep : sig
    val adams_bashforth : step_config -> SDE.t -> Tensor.t -> 
      Tensor.t array -> Tensor.t
    val adams_moulton : step_config -> SDE.t -> Tensor.t -> 
      Tensor.t array -> Tensor.t
    val bdf : step_config -> SDE.t -> Tensor.t -> 
      Tensor.t array -> Tensor.t
  end

  module PathIntegral : sig
    type integral_method = 
      | Trapezoidal 
      | Simpson 
      | GaussKronrod 
      | AdaptiveQuad

    val integrate : integral_method -> Tensor.t -> float array -> float
  end
end