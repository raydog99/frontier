open Torch

module Solver : sig
  type t = [
    | `Euler  (** Simple Euler integration *)
    | `RK4    (** 4th order Runge-Kutta *)
    | `AdaptiveHeun of {rtol: float; atol: float}  (** Adaptive Heun method *)
    | `DormandPrince of {rtol: float; atol: float} (** Dormand-Prince (DOPRI5) *)
  ]
end

type parameters = float array  (** θ ∈ Θ ⊆ Rk *)
  
type state = {
    u: Tensor.t;  (** Main state variable *)
    v: Tensor.t option;  (** Optional second variable for systems *)
    spatial_dims: int * int;  (** Dimensions of spatial domain *)
  }

type scattering_config = {
    orders: int;          (** Maximum scattering order *)
    orientations: int;    (** Number of orientations *)
    scales: int;          (** Number of scales *)
  }

type node_config = {
    hidden_dims: int array;  (** Layer dimensions *)
    activation: Tensor.t -> Tensor.t;  (** Activation function *)
    dt: float;              (** Default time step *)
    solver: Solver.t;       (** Integration method *)
    dropout_rate: float;    (** Dropout regularization *)
}

val make_gaussian_filter : sigma:float -> size:int -> Tensor.t
(** Creates a Gaussian lowpass filter *)

val make_morlet_filter : scale:float -> orientation:float -> size:int -> Tensor.t
(** Creates a Morlet wavelet filter *)

val laplacian : Tensor.t -> Tensor.t
(** Computes discrete Laplacian operator *)

module ScatteringTransform : sig
  type t 
  (** Abstract type for scattering transform *)

  type selection_strategy = {
    energy_threshold: float;    (** Minimum energy to preserve *)
    correlation_threshold: float;  (** Maximum correlation between coefficients *)
    max_coefficients: int;     (** Maximum number of coefficients to select *)
  }

  val init_filters : size:int -> orientations:int -> scales:int -> t
  (** Initialize scattering filters *)

  val forward : t -> Tensor.t -> Tensor.t
  (** Apply scattering transform *)

  val select_coefficients : Tensor.t list -> selection_strategy -> Tensor.t list
  (** Select most informative coefficients *)
end

module NeuralODE : sig
  type t
  (** Abstract type for neural ODE *)

  val make : config:node_config -> augment_dim:int option -> 
            regularization:float -> t
  (** Create neural ODE with given configuration *)

  val solve : t -> Tensor.t -> float -> float -> ?dt:float -> Tensor.t list
  (** [solve node x t0 t1 ~dt] Solves NODE from t0 to t1 with optional step size dt *)

  val regularization_divergence : t -> Tensor.t
  (** Compute regularization divergence *)
end

module DivergenceMeasures : sig
  type config = {
    state_weight: float;        (** Weight for state matching divergence *)
    deriv_weight: float;        (** Weight for derivative matching divergence *)
    spectral_weight: float option;  (** Optional weight for spectral regularization *)
    phase_weight: float option;     (** Optional weight for phase space regularization *)
    l1_weight: float option;        (** Optional L1 regularization weight *)
    l2_weight: float option;        (** Optional L2 regularization weight *)
    reconstruction_weight: float option;  (** Optional reconstruction divergence weight *)
  }

  val temporal_regression : pred:Tensor.t -> target:Tensor.t ->
                          pred_deriv:Tensor.t -> target_deriv:Tensor.t ->
                          beta:float -> Tensor.t
  (** Compute temporal regression divergence *)

  val spectral_regularization : Tensor.t list -> Tensor.t
  (** Compute spectral regularization *)

  val phase_space_regularization : Tensor.t list -> Tensor.t
  (** Compute phase space regularization *)

  val compute_divergence : config:config -> pred:Tensor.t -> 
                    target:Tensor.t -> pred_deriv:Tensor.t ->
                    target_deriv:Tensor.t -> model:NeuralODE.t ->
                    trajectories:Tensor.t list -> Tensor.t
  (** Compute total divergence with all components *)
end

module TRENDy : sig
  type t
  (** Abstract type for TRENDy model *)

  val make : scattering_config:scattering_config -> 
            node_config:node_config -> t
  (** Create TRENDy model *)

  val compute_effective_state : t -> state -> Tensor.t
  (** Map state to effective dynamics space *)

  val predict : t -> state -> parameters -> Tensor.t list
  (** [predict model state params] Predicts trajectory from initial state 
      @return List of states in effective dynamics space *)

  val train : t -> dataset:(state * parameters) list ->
             learning_rate:float -> n_epochs:int -> t
  (** [train model dataset ~learning_rate ~n_epochs] Trains model on dataset
      @return Updated model *)
end

module PatternAnalysis : sig
  type pattern_type = 
    | Homogeneous   (** No spatial pattern *)
    | Stripes       (** Stripe patterns *)
    | SparseSpots   (** Isolated spots *)
    | DenseSpots    (** Dense spot patterns *)
    | Mixed         (** Mixed patterns *)

  val analyze_frequencies : state -> float array
  (** [analyze_frequencies state] Computes spatial frequency spectrum *)

  val classify_pattern : state -> pattern_type
  (** [classify_pattern state] Classifies spatial pattern type *)

  val analyze_evolution : state list -> pattern_type option
  (** [analyze_evolution trajectory] Analyzes pattern evolution over time
      @return Some pattern_type if consistent pattern emerges, None if inconsistent *)
end

module DynamicalSystems : sig
  module GrayScott : sig
    type parameters = {
      F: float;  (** Feed rate *)
      k: float;  (** Kill rate *)
      Du: float; (** Diffusion rate for u *)
      Dv: float; (** Diffusion rate for v *)
    }

    val init_state : size:int -> params:parameters -> state
    (** Initialize state with random perturbation *)

    val step : state -> parameters -> float -> state
    (** Single evolution step *)

    val evolve : state -> parameters -> n_steps:int -> dt:float -> state list
    (** Evolve system for multiple steps *)

    val detect_bifurcation : state -> parameters -> bool
    (** Detect Turing bifurcation *)
  end

  module Brusselator : sig
    type parameters = {
      A: float;  (** Input parameter *)
      B: float;  (** Control parameter *)
      Du: float; (** Diffusion coefficient for u *)
      Dv: float; (** Diffusion coefficient for v *)
    }

    type oscillation_mode = {
      frequency: float;      (** Oscillation frequency *)
      amplitude: float;      (** Oscillation amplitude *)
      wavevector: float * float;  (** Spatial wave vector (kx, ky) *)
      phase: float;         (** Phase angle *)
    }

    val init_state : size:int -> params:parameters -> state
    (** Initialize state near equilibrium *)

    val step : state -> parameters -> float -> state
    (** Single evolution step *)

    val evolve : state -> parameters -> n_steps:int -> dt:float -> state list
    (** Evolve system for multiple steps *)

    val analyze_oscillations : state list -> oscillation_mode
    (** Analyze oscillation characteristics *)

    val detect_hopf_bifurcation : state list -> parameters -> bool
    (** Detect Hopf bifurcation *)
  end
end