open Torch

module Complex : sig
  type t = {re: Tensor.t; im: Tensor.t}
  val create : Tensor.t -> Tensor.t -> t
  val zero : unit -> t
  val add : t -> t -> t
  val mul : t -> t -> t
  val conj : t -> t
  val abs_sq : t -> Tensor.t
  val scale : t -> float -> t
  val inverse : t -> t
end

module MatrixOps : sig
  type matrix_properties = {
    condition_number: float;
    rank: int;
    is_positive_definite: bool
  }
  
  val check_stability : Tensor.t -> matrix_properties
  val robust_inverse : Tensor.t -> Tensor.t
  val solve_symmetric_system : Tensor.t -> Tensor.t -> Tensor.t
end

module SpectralEstimation : sig
  type spectral_matrix = {
    frequencies: Tensor.t;
    density: Complex.t array array;
    coherence: Tensor.t array array option;
    phase: Tensor.t array array option;
    error_bounds: Complex.t array array option;
  }
  
  val compute_autocovariance_matrix : Tensor.t -> int -> Tensor.t array array
  val estimate_spectral_density_matrix : Tensor.t -> 'a -> spectral_matrix
end

module ARMAModel : sig
  type model_params = {
    ar_coeffs: Tensor.t;
    ma_coeffs: Tensor.t;
    innovation_var: float;
  }

  type parameter_constraints = {
    ar_stationary: bool;
    ma_invertible: bool;
    max_ar_coef: float;
    max_ma_coef: float;
    min_variance: float;
  }

  val create : int -> int -> model_params
  val check_stationarity : Tensor.t -> bool
  val check_invertibility : Tensor.t -> bool
  val enforce_constraints : model_params -> parameter_constraints -> model_params
  val spectral_density : model_params -> Tensor.t -> Complex.t
  val simulate : model_params -> int -> Tensor.t
end

module ParameterEstimation : sig
  type estimation_config = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    momentum: float;
    batch_size: int option;
  }

  val default_config : estimation_config
  val estimate_ar : Tensor.t -> int -> Tensor.t * float
  val estimate_ma : Tensor.t -> int -> Tensor.t * float
  val optimize_parameters : (Tensor.t -> float * Tensor.t) -> Tensor.t -> estimation_config -> Tensor.t
  val estimate_arma : Tensor.t -> int -> int -> estimation_config -> Tensor.t
end

module KalmanFilter : sig
  type filter_state = {
    mean: Tensor.t;
    covariance: Tensor.t;
    gain: Tensor.t;
    innovation: Tensor.t;
    innovation_cov: Tensor.t;
    loglik: float;
  }

  type system_matrices = {
    transition: Tensor.t;
    observation: Tensor.t;
    system_noise: Tensor.t;
    observation_noise: Tensor.t;
  }

  val predict_step : filter_state -> system_matrices -> filter_state
  val update_step : filter_state -> Tensor.t -> system_matrices -> filter_state
  val smooth_state : filter_state array -> system_matrices -> filter_state array
end

module EMAlgorithm : sig
  type em_state = {
    params: ARMAModel.model_params;
    filtered_states: KalmanFilter.filter_state array;
    smoothed_states: KalmanFilter.filter_state array;
    loglik: float;
    iteration: int;
    converged: bool;
  }

  val create_initial_state : Tensor.t -> em_state
  val e_step : em_state -> Tensor.t -> KalmanFilter.system_matrices -> em_state
  val m_step : em_state -> Tensor.t -> em_state
  val check_convergence : float -> float -> float -> bool
  val run : ?max_iter:int -> ?epsilon:float -> Tensor.t -> em_state
end