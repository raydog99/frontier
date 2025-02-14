open Torch

module Filtration : sig
  type t = {
    time_points: float array;
    sub_sigma_fields: (float array) array;
  }
  
  val create : float array -> t
  val is_adapted : t -> float array array -> bool
end

module Measure : sig
  type t = {
    domain: float * float;
    density: float -> float;
    total_mass: float option;
  }
  
  val lebesgue : float * float -> t
  val counting : float array -> t
  val integrate : t -> (float -> float) -> float
  val is_finite : t -> bool
end

module RandomMeasure : sig
  type t = {
    intensity: Measure.t;
    generator: unit -> float;
  }
  
  val poisson : Measure.t -> t
  val simulate : t -> float -> int -> float array
  val compensate : t -> t
end

module StochasticIntegral : sig
  val brownian_motion : float -> int -> float -> float array
  val ito_integral : float array -> float array -> float -> float -> float
  val poisson_integral : float array -> RandomMeasure.t -> float -> float
end

module Space : sig
  type t = {
    sample_space: Measure.t;
    filtration: Filtration.t;
    probability: float -> float;
  }
  
  val create : Measure.t -> Filtration.t -> (float -> float) -> t
  val expectation : t -> (float -> float) -> float
  val conditional_expectation : t -> (float -> float) -> (float -> bool) -> float
end

module Process : sig
  type 'a t = {
    paths: ('a array) array;
    time_points: float array;
    filtration: Filtration.t;
  }
  
  val create : ('a array) array -> float array -> 'a t
  val is_martingale : float t -> Space.t -> bool
  val quadratic_variation : float t -> float array
  val is_continuous : float t -> float -> bool
end

module LevyProcess : sig
  type t = float Process.t
  
  val create : float array -> float array -> t
end

module Branching : sig
  type parameters = {
    b: float;
    sigma: float;
    pi: Measure.t;
  }
  
  val phi : parameters -> float -> float
  val is_subcritical : parameters -> bool
  val compute_domain : parameters -> float
end

module Immigration : sig
  type parameters = {
    beta: float;
    nu: Measure.t;
  }
  
  val psi : parameters -> float -> float
  val has_finite_mean : parameters -> bool
end

module Process : sig
  type parameters = {
    branching: Branching.parameters;
    immigration: Immigration.parameters;
    x0: float;
  }
  
  module ODESolver : sig
    type solution = {
      times: float array;
      values: float array;
    }
    
    val solve : parameters -> Complex.t -> float -> float -> solution
  end
  
  val transition_probability : parameters -> float -> float -> float -> float
  val laplace_transform : parameters -> float -> float -> float
  val characteristic_function : parameters -> float -> float -> Complex.t
  val compute_lifetime : parameters -> float -> float -> float
  val simulate : parameters -> float -> float -> int -> Tensor.t list
end

module MultiCurveModel : sig
  type model_parameters = {
    ell: float -> float;
    lambda: Tensor.t;
    c: int -> float -> float;
    gamma: int -> Tensor.t;
    cbi_params: Process.parameters;
  }
  
  module TermStructure : sig
    type forward_curve = {
      tenors: float array;
      rates: float array;
      interpolator: float -> float;
    }
    
    val create_forward_curve : float array -> float array -> forward_curve
    val discount_factor : forward_curve -> float -> float -> float
  end
  
  module ForwardRates : sig
    val forward_rate : model_parameters -> float -> float -> float -> Tensor.t -> float
    val short_rate : model_parameters -> float -> Tensor.t -> float
    val log_spread : model_parameters -> int -> float -> Tensor.t -> float
  end
  
  module Spreads : sig
    val spot_spread : model_parameters -> float -> float -> Tensor.t -> float
    val forward_spread : model_parameters -> float -> float -> float -> Tensor.t -> float
    val check_monotonicity : model_parameters -> bool
  end
end

module Pricing : sig
  type pricing_result = {
    price: float;
    delta: float option;
    gamma: float option;
    vega: float option;
    error_bound: float option;
  }
  
  module FFT : sig
    type grid_params = {
      n_points: int;
      eta: float;
      alpha: float;
    }
    
    val modified_char_fn : 
      MultiCurveModel.model_parameters -> 
      float -> float -> float -> float -> Tensor.t -> Complex.t
    
    val create_grid : grid_params -> float array * float array
    val compute_fft_inputs : 
      MultiCurveModel.model_parameters -> 
      float -> float -> float -> grid_params -> Complex.t array
    val fft : Complex.t array -> Complex.t array
    val price_caplet : 
      MultiCurveModel.model_parameters -> 
      float -> float -> float -> float -> float
  end
  
  module Quantization : sig
    type grid = {
      points: float array;
      weights: float array;
    }
    
    val lp_distance : grid -> float -> float -> float
    val optimize_grid : grid -> float -> float -> int -> grid
    val companion_weights : 
      grid -> MultiCurveModel.model_parameters -> float -> float array
    val price_caplet : 
      MultiCurveModel.model_parameters -> 
      float -> float -> float -> float -> float
  end
end

module Calibration : sig
  type market_caplet = {
    tenor: float;
    maturity: float;
    strike: float;
    price: float;
    bid: float option;
    ask: float option;
  }
  
  type market_data = {
    observation_date: float;
    ois_curve: MultiCurveModel.TermStructure.forward_curve;
    caplets: market_caplet array;
    tenors: float array;
  }
  
  module Divergence : sig
    type weights = {
      price_weight: float;
      spread_weight: float;
      regularization: float;
    }
    
    val price_divergence : 
      MultiCurveModel.model_parameters -> market_data -> float
    val spread_divergence : 
      MultiCurveModel.model_parameters -> market_data -> float
    val regularization : 
      MultiCurveModel.model_parameters -> float
    val total_divergence : 
      MultiCurveModel.model_parameters -> market_data -> weights -> float
  end
  
  module Constraints : sig
    type bounds = {
      lower: float;
      upper: float;
    }
    
    type parameter_constraints = {
      alpha: bounds;
      theta: bounds;
      b: bounds;
      sigma: bounds;
      eta: bounds;
    }
    
    val default_constraints : parameter_constraints
    val apply_constraints : 
      Process.parameters -> parameter_constraints -> Process.parameters
  end
  
  module Optimization : sig
    type config = {
      learning_rate: float;
      max_iterations: int;
      tolerance: float;
      momentum: float;
    }
    
    val optimize : 
      Process.parameters -> 
      market_data -> 
      Divergence.weights -> 
      Constraints.parameter_constraints -> 
      config -> 
      Process.parameters
  end
end