open Torch

(* Core types *)
type term_structure = {
  term: float;
  yield: float;
}

type yield_curve = {
  date: float;  (* Unix timestamp *)
  points: term_structure array;
}

type nelson_siegel_params = {
  beta1: float;  (* Level *)
  beta2: float;  (* Slope *)
  beta3: float;  (* Curvature *)
  lambda: float; (* Decay rate *)
}

type integration_method = 
  | Trapezoidal
  | Simpson
  | RectangularLeft
  | RectangularRight

type basis_type =
  | FourierBasis of {
      max_freq: int;
      period: float;
    }
  | NelsonSiegelBasis
  | ExponentialBasis of {
      n_terms: int;
      rate: float array;
    }
  | GaussianBasis of {
      n_terms: int;
      centers: float array;
      width: float;
    }

(* Core functions *)
val create_from_float_array : float array -> Tensor.t
val tensor_to_float_array : Tensor.t -> float array
val zeros : int list -> Tensor.t
val ones : int list -> Tensor.t
val randn : int list -> mean:float -> std:float -> Tensor.t

(* Nelson-Siegel functions *)
val nelson_siegel_basis : float -> float -> float array
val nelson_siegel_yield : nelson_siegel_params -> float -> float
val estimate_nelson_siegel_params : term_structure array -> nelson_siegel_params

(* RobustMatrix *)
module RobustMatrix : sig
  val estimate_condition_number : Tensor.t -> float
  val modified_gram_schmidt : Tensor.t -> Tensor.t * Tensor.t
  val robust_cholesky : Tensor.t -> float -> Tensor.t * Tensor.t
end

(* Optimization *)
module Optimization : sig
  type optimizer_config = {
    method_type: [`Adam | `LBFGS | `SGD];
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    momentum: float option;
    beta1: float;
    beta2: float;
  }

  val default_optimizer_config : optimizer_config

  val adam : f:(Tensor.t -> float * Tensor.t) -> init:Tensor.t -> 
    config:optimizer_config -> Tensor.t

  val lbfgs : f:(Tensor.t -> float * Tensor.t) -> init:Tensor.t -> 
    config:optimizer_config -> Tensor.t
end

(* D-Operator functions *)
val derivative : (float -> float) -> float -> float -> float
val second_derivative : (float -> float) -> float -> float -> float
val d_operator_matrix : term_structure array -> float -> Tensor.t
val d2_operator_matrix : term_structure array -> float -> Tensor.t
val integrate : (float -> float) -> float -> float -> int -> integration_method -> float
val compute_penalty_matrix : term_structure array -> float -> Tensor.t

(* Basis functions *)
val fourier_basis : int -> float -> float -> float array
val exponential_basis : float array -> float -> float array
val gaussian_basis : float array -> float -> float -> float array
val approximate_function : (float -> float array) -> float array -> float -> float

(* Gaussian Process *)
module GP : sig
  type kernel_type =
    | RBF
    | Linear
    | Periodic of float
    | Composite of kernel_type list * [`Sum | `Product]

  type kernel_params = {
    base_length_scale: float;
    base_signal_variance: float;
    noise_variance: float;
    time_decay: float option;
    amplitude_growth: float option;
  }

  type gp_model = {
    params: kernel_params;
    kernel_type: kernel_type;
    mean: Tensor.t option;
  }

  val create_model : ?mean:Tensor.t option -> kernel_params -> gp_model
  
  val compute_kernel_matrix : Tensor.t -> Tensor.t -> kernel_params -> 
    kernel_type -> float -> Tensor.t

  val predict : gp_model -> Tensor.t -> Tensor.t -> Tensor.t -> float -> 
    Tensor.t * Tensor.t

  val compute_marginal_likelihood : gp_model -> Tensor.t -> Tensor.t -> float
end

(* Dynamic GP *)
module DynamicGP : sig
  type dynamic_gp_state = {
    model: GP.gp_model;
    posterior_mean: Tensor.t option;
    posterior_cov: Tensor.t option;
  }

  val create_initial_state : GP.gp_model -> dynamic_gp_state
  val update_state : dynamic_gp_state -> Tensor.t option -> dynamic_gp_state
  val predict : dynamic_gp_state -> Tensor.t -> float -> Tensor.t * Tensor.t
end

(* Time Series *)
module TimeSeries : sig
  type var_model = {
    coefficients: Tensor.t array;
    intercept: Tensor.t;
  }

  val fit_var_model : yield_curve array -> int -> var_model
  val forecast : var_model -> yield_curve array -> Tensor.t
  val compute_information_criteria : var_model -> yield_curve array -> int -> 
    float * float * float
end

(* State Space *)
module StateSpace : sig
  type state_space_model = {
    transition_matrix: Tensor.t;
    observation_matrix: Tensor.t;
    state_noise_cov: Tensor.t;
    obs_noise_cov: Tensor.t;
    initial_state: Tensor.t;
    initial_state_cov: Tensor.t;
  }

  type kalman_state = {
    pred_state: Tensor.t;
    pred_cov: Tensor.t;
    filtered_state: Tensor.t;
    filtered_cov: Tensor.t;
    log_likelihood: float;
  }

  val create_dns_model : nelson_siegel_params -> term_structure array -> state_space_model
  val predict_step : state_space_model -> kalman_state -> kalman_state
  val update_step : state_space_model -> kalman_state -> Tensor.t -> kalman_state
  val kalman_filter : state_space_model -> float array array -> kalman_state
end

(* Parameter Learning *)
module ParameterLearning : sig
  val estimate_ols : Tensor.t -> Tensor.t -> Tensor.t
  val estimate_penalized_ls : Tensor.t -> Tensor.t -> float -> Tensor.t -> Tensor.t
  val estimate_bayesian : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> 
    float -> Tensor.t * Tensor.t
  val cross_validate_parameters : Tensor.t -> Tensor.t -> int -> 
    (Tensor.t -> Tensor.t -> Tensor.t) -> float
end