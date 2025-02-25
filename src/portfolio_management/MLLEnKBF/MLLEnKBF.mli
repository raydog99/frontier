open Torch

(* Kalman-Bucy *)
module KalmanBucy : sig
  (* Model parameters for Kalman-Bucy *)
  type model_params = {
    a : Tensor.t;  (* Square dx x dx matrix - drift term *)
    c : Tensor.t;  (* dy x dx matrix - observation matrix *)
    r1 : Tensor.t; (* Square matrix - signal noise covariance *)
    r2 : Tensor.t; (* Square matrix - observation noise covariance *)
    m0 : Tensor.t; (* Initial mean *)
    p0 : Tensor.t; (* Initial covariance *)
  }

  (* [ricc params q] computes the Riccati drift function. *)
  val ricc : model_params -> Tensor.t -> Tensor.t

  (* [kalman_bucy_update params m p dt y_old y_new] performs one step of the Kalman-Bucy update. *)
  val kalman_bucy_update : model_params -> Tensor.t -> Tensor.t -> int -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t

  (* [kalman_filter params observations n_steps dt] runs the Kalman filter. *)
  val kalman_filter : model_params -> Tensor.t array -> int -> int -> Tensor.t list

  (* [mkv_vanilla_step params x m p dt dw y_old y_new] performs one step of the vanilla McKean-Vlasov process. *)
  val mkv_vanilla_step : model_params -> Tensor.t -> Tensor.t -> Tensor.t -> int -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

  (* [mkv_deterministic_step params x m p dt dw y_old y_new] performs one step of the deterministic McKean-Vlasov process. *)
  val mkv_deterministic_step : model_params -> Tensor.t -> Tensor.t -> Tensor.t -> int -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

  (* [mkv_deterministic_transport_step params x m p dt y_old y_new] performs one step of the deterministic transport McKean-Vlasov process. *)
  val mkv_deterministic_transport_step : model_params -> Tensor.t -> Tensor.t -> Tensor.t -> int -> Tensor.t -> Tensor.t -> Tensor.t
end

(* Localization Functions *)
module Localization : sig
  (* Localization functions *)
  type localization_function =
    | Uniform     (* Uniform localization *)
    | Triangular  (* Triangular localization *)
    | GaspariCohn (* Gaspari-Cohn localization *)

  (* [uniform_localization d r] computes the uniform localization function. *)
  val uniform_localization : float -> float -> float

  (* [triangular_localization d r] computes the triangular localization function. *)
  val triangular_localization : float -> float -> float

  (* [gaspari_cohn_localization d r] computes the Gaspari-Cohn localization function. *)
  val gaspari_cohn_localization : float -> float -> float

  (* [get_localization_function loc_fn] returns the implementation of the specified localization function. *)
  val get_localization_function : localization_function -> (float -> float -> float)

  (* [visualize_localization_function loc_fn r n_points] visualizes the localization function. *)
  val visualize_localization_function : localization_function -> float -> int -> float list * float list

  (* [apply_localization cov loc_fn radius] applies localization to a covariance matrix. *)
  val apply_localization : Tensor.t -> localization_function -> int -> Tensor.t

  (* [apply_localized_update particles means cov_localized gain innovations dt] applies localized update to particles. *)
  val apply_localized_update : Tensor.t list -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t list -> int -> Tensor.t list
end

(* Ensemble Kalman-Bucy Filter *)
module EnKBF : sig
  (* EnKBF variants *)
  type enkbf_variant =
    | Vanilla                (* VEnKBF - with perturbed observations *)
    | Deterministic          (* DEnKBF - without perturbed observations *)
    | DeterministicTransport (* DTEnKBF - completely deterministic *)

  (* [compute_mean ensemble] computes the mean of an ensemble. *)
  val compute_mean : Tensor.t list -> Tensor.t

  (* [compute_covariance ensemble mean] computes the covariance of an ensemble. *)
  val compute_covariance : Tensor.t list -> Tensor.t -> Tensor.t

  (* [vanilla_enkbf_step params particle mean cov dt dw dy] performs one step of the vanilla EnKBF. *)
  val vanilla_enkbf_step : KalmanBucy.model_params -> Tensor.t -> Tensor.t -> Tensor.t -> int -> Tensor.t -> Tensor.t -> Tensor.t

  (* [deterministic_enkbf_step params particle mean cov dt dw dy] performs one step of the deterministic EnKBF. *)
  val deterministic_enkbf_step : KalmanBucy.model_params -> Tensor.t -> Tensor.t -> Tensor.t -> int -> Tensor.t -> Tensor.t -> Tensor.t

  (* [deterministic_transport_enkbf_step params particle mean cov dt dy] performs one step of the deterministic transport EnKBF. *)
  val deterministic_transport_enkbf_step : KalmanBucy.model_params -> Tensor.t -> Tensor.t -> Tensor.t -> int -> Tensor.t -> Tensor.t

  (* [enkbf_step params variant ensemble_i ensemble_mean ensemble_cov dt dw dy] performs one step of the EnKBF for the specified variant. *)
  val enkbf_step : KalmanBucy.model_params -> enkbf_variant -> Tensor.t -> Tensor.t -> Tensor.t -> int -> Tensor.t -> Tensor.t -> Tensor.t

  (* [discretized_enkbf_step params variant ensemble_i ensemble_mean ensemble_cov dt k y_data] performs one discretized step of the EnKBF. *)
  val discretized_enkbf_step : KalmanBucy.model_params -> enkbf_variant -> Tensor.t -> Tensor.t -> Tensor.t -> int -> int -> Tensor.t array -> Tensor.t

  (* [run_enkbf params variant n_particles n_steps dt y_data] runs the EnKBF algorithm. *)
  val run_enkbf : KalmanBucy.model_params -> enkbf_variant -> int -> int -> int -> Tensor.t array -> Tensor.t list

  (* [run_localized_enkbf params variant n_particles n_steps dt y_data loc_fn loc_radius] runs the localized EnKBF algorithm. *)
  val run_localized_enkbf : KalmanBucy.model_params -> enkbf_variant -> int -> int -> int -> Tensor.t array -> Localization.localization_function -> int -> Tensor.t list * Tensor.t list
end

(* Multilevel Monte Carlo *)
module MLMC : sig
  (* [run_single_level params variant n_particles level_l n_steps dt y_data loc_fn loc_radius] runs a single level of the MLMC estimator. *)
  val run_single_level : KalmanBucy.model_params -> EnKBF.enkbf_variant -> int -> int -> int -> int -> Tensor.t array -> Localization.localization_function -> int -> Tensor.t list

  (* [run_coupled_levels params variant n_particles level_l n_steps dt y_data loc_fn loc_radius] runs coupled levels of the MLMC estimator. *)
  val run_coupled_levels : KalmanBucy.model_params -> EnKBF.enkbf_variant -> int -> int -> int -> int -> Tensor.t array -> Localization.localization_function -> int -> Tensor.t list * Tensor.t list

  (* [multilevel_estimate params variant phi max_level n_particles_list n_steps dt y_data loc_fn loc_radius] computes the multilevel estimator. *)
  val multilevel_estimate : KalmanBucy.model_params -> EnKBF.enkbf_variant -> (Tensor.t -> Tensor.t) -> int -> int list -> int -> int -> Tensor.t array -> Localization.localization_function -> int -> Tensor.t

  (* [localized_multilevel_estimate params variant phi max_level n_particles_list n_steps dt y_data loc_fn loc_radius] computes the localized multilevel estimator. *)
  val localized_multilevel_estimate : KalmanBucy.model_params -> EnKBF.enkbf_variant -> (Tensor.t -> Tensor.t) -> int -> int list -> int -> int -> Tensor.t array -> Localization.localization_function -> int -> Tensor.t
end

(* Normalizing Constant Estimation *)
module NCE : sig
  (* [estimate_normalizing_constant params means observations dt level] estimates the normalizing constant. *)
  val estimate_normalizing_constant : KalmanBucy.model_params -> Tensor.t list -> Tensor.t array -> int -> int -> Tensor.t

  (* [estimate_log_normalizing_constant params means observations dt level] estimates the log normalizing constant. *)
  val estimate_log_normalizing_constant : KalmanBucy.model_params -> Tensor.t list -> Tensor.t array -> int -> int -> Tensor.t

  (* [estimate_loc_normalizing_constant params means covariances observations dt level loc_fn loc_radius] estimates the localized normalizing constant. *)
  val estimate_loc_normalizing_constant : KalmanBucy.model_params -> Tensor.t list -> Tensor.t list -> Tensor.t array -> int -> int -> Localization.localization_function -> int -> Tensor.t

  (* [multilevel_estimate params variant max_level n_particles_list n_steps dt observations loc_fn loc_radius] computes the multilevel normalizing constant. *)
  val multilevel_estimate : KalmanBucy.model_params -> EnKBF.enkbf_variant -> int -> int list -> int -> int -> Tensor.t array -> Localization.localization_function -> int -> Tensor.t

  (* [multilevel_log_estimate params variant max_level n_particles_list n_steps dt observations loc_fn loc_radius] computes the multilevel log normalizing constant. *)
  val multilevel_log_estimate : KalmanBucy.model_params -> EnKBF.enkbf_variant -> int -> int list -> int -> int -> Tensor.t array -> Localization.localization_function -> int -> Tensor.t
end

(* Multilevel Localized Ensemble Kalman-Bucy Filter *)
module MLLEnKBF : sig
  (* Tracking state and history *)
  type 'a history = {
    trajectories: 'a list list;   (* List of particle lists at each time step *)
    means: Tensor.t list;         (* Mean at each time step *)
    covariances: Tensor.t list;   (* Covariance at each time step *)
    log_norm_constants: float list; (* Log normalizing constants *)
  }

  (* Configuration for the MLLEnKBF algorithm *)
  type config = {
    max_level: int;                (* Maximum level L *)
    start_level: int;              (* Starting level l* *)
    n_particles: int list;         (* Number of particles at each level *)
    n_steps: int;                  (* Number of time steps *)
    dt: int;                       (* Base time step *)
    variant: EnKBF.enkbf_variant;  (* EnKBF variant to use *)
    loc_function: Localization.localization_function; (* Localization function *)
    loc_radius: int;               (* Localization radius *)
  }

  (* [generate_observations params n_steps dt] generates synthetic observations for a model. *)
  val generate_observations : KalmanBucy.model_params -> int -> int -> Tensor.t array

  (* [run_algorithm1 params config observations phi] (Localized Multilevel Ensemble Kalman-Bucy Filter). *)
  val run_algorithm1 : KalmanBucy.model_params -> config -> Tensor.t array -> (Tensor.t -> Tensor.t) -> Tensor.t

  (* [run_algorithm2 params config observations] (Localized Multilevel Estimation of Normalizing Constants). *)
  val run_algorithm2 : KalmanBucy.model_params -> config -> Tensor.t array -> Tensor.t

  (* [run_algorithm2_log params config observations] for log normalizing constant estimation. *)
  val run_algorithm2_log : KalmanBucy.model_params -> config -> Tensor.t array -> Tensor.t
end

(* Parameter Estimation *)
module ParameterEstimation : sig
  (* Parameter estimation configuration *)
  type config = {
    params: KalmanBucy.model_params;  (* Initial model parameters *)
    mllenk_config: MLLEnKBF.config;   (* Configuration for MLLEnKBF *)
    param_idx: int list;              (* Indices of parameters to estimate *)
    param_bounds: (float * float) list; (* Min and max bounds for each parameter *)
    n_iterations: int;                (* Number of optimization iterations *)
    learning_rate: float;             (* Learning rate for optimization *)
  }

  (* [extract_params params indices] extracts parameter values from a model. *)
  val extract_params : KalmanBucy.model_params -> int list -> float list

  (* [update_params params indices values] updates a model with new parameter values. *)
  val update_params : KalmanBucy.model_params -> int list -> float list -> KalmanBucy.model_params

  (* [log_norm_constant_objective params config observations] computes the log normalizing constant objective function. *)
  val log_norm_constant_objective : KalmanBucy.model_params -> config -> Tensor.t array -> float

  (* [differential_evolution config observations] performs parameter estimation using differential evolution. *)
  val differential_evolution : config -> Tensor.t array -> KalmanBucy.model_params

  (* [optimize_simple config observations] performs parameter estimation using a simple gradient-free method. *)
  val optimize_simple : config -> Tensor.t array -> KalmanBucy.model_params
end