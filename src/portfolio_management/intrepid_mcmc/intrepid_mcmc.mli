open Torch

(** Core configuration and state types *)
type proposal_config = {
  exploration_ratio : float;
  n_dims : int;
  anchor_point : Tensor.t;
  proposal_scale : float;
  angular_scales : float array;
  radial_gamma : float;
  min_accept_rate : float;
  max_accept_rate : float;
}

type mcmc_state = {
  current_point : Tensor.t;
  current_value : float;
  accepted : bool;
  acceptance_count : int;
  total_count : int;
  parent_density : float;
  transformed_value : float;
}

type distribution_shape =
  | Radially_Symmetric
  | Uniform_Shape of float * float
  | Unimodal_Convex
  | General_Shape

type density_type =
  | Parent
  | Target
  | Transformation

type proposal_type =
  | Local of float
  | Global of float * float array
  | Adaptive of float * float

type sampling_stats = {
  acceptance_rate : float;
  effective_sample_size : float;
  mean : Tensor.t;
  covariance : Tensor.t;
}

type chain_config = {
  n_samples : int;
  n_chains : int;
  n_dims : int;
  burn_in : int;
  thin : int;
  adaptation_window : int;
}

(** Core distribution functions *)
val standard_normal : Tensor.t -> float
val multivariate_normal : ?mu:Tensor.t option -> ?sigma:Tensor.t option -> Tensor.t -> float
val uniform : ?lower:Tensor.t option -> ?upper:Tensor.t option -> Tensor.t -> float
val mixture : float list -> (Tensor.t -> float) list -> Tensor.t -> float
val gumbel : ?mu:float -> ?beta:float -> Tensor.t -> float

(** Utility functions *)
val to_spherical : Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
val from_spherical : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
val log_uniform : unit -> Tensor.t
val effective_sample_size : Tensor.t list -> float
val compute_stats : Tensor.t list -> sampling_stats
val stable_log_sum_exp : float list -> float
val numerically_stable_ratio : float -> float -> float

(** Numerical stability *)
module NumericalStability : sig
  type numerical_params = {
    epsilon : float;
    max_condition : float;
    min_eigenvalue : float;
    max_gradient : float;
  }

  val default_params : numerical_params

  type stability_check = {
    condition_number : float;
    eigenvalue_range : float * float;
    gradient_norm : float;
    has_nans : bool;
  }

  val check_matrix_stability : Tensor.t -> numerical_params -> bool
  val stabilize_covariance : Tensor.t -> numerical_params -> Tensor.t
  val compute_stable_gradient : (Tensor.t -> float) -> Tensor.t -> numerical_params -> Tensor.t
  val check_numerical_stability : Tensor.t -> (Tensor.t -> float) -> numerical_params -> stability_check
end

(** Complete RTF *)
module CompleteRTF : sig
  type rtf_form = 
    | RadiallySymmetric
    | UniformBounds
    | UnimodalConvex
    | GeneralForm of {
        transform : Tensor.t -> Tensor.t;
        derivative : Tensor.t -> Tensor.t;
        exists : Tensor.t -> bool;
        contour_map : Tensor.t -> float;
      }

  val check_convexity : Tensor.t list -> (Tensor.t -> float) -> bool
  val construct_radially_symmetric_rtf : Tensor.t -> rtf_form
  val construct_uniform_rtf : float * float -> rtf_form
  val construct_unimodal_rtf : (Tensor.t -> float) -> Tensor.t -> rtf_form
  val apply_rtf : rtf_form -> Tensor.t -> Tensor.t
  val get_derivative : rtf_form -> Tensor.t -> Tensor.t
  val exists : rtf_form -> Tensor.t -> bool
  val get_contour_value : rtf_form -> Tensor.t -> float
end

(** Coordinate system *)
module CoordinateSystem : sig
  type spherical_coords = {
    r : Tensor.t;
    theta : Tensor.t array;
  }

  val cartesian_to_spherical : Tensor.t -> Tensor.t -> spherical_coords
  val spherical_to_cartesian : spherical_coords -> Tensor.t -> Tensor.t
  val jacobian_spherical_to_cartesian : spherical_coords -> float
end

(** Kernel *)
module Kernel : sig
  type kernel_type =
    | Local 
    | Intrepid
    | Mixed of float

  type transition_kernel = {
    name : string;
    kernel_type : kernel_type;
    generating_function : Tensor.t -> Tensor.t -> float;
    acceptance_probability : Tensor.t -> Tensor.t -> float;
    is_reversible : bool;
  }

  val make_local_kernel : float -> transition_kernel
  val make_intrepid_kernel : proposal_config -> CompleteRTF.rtf_form -> transition_kernel
  val compose_kernels : transition_kernel -> transition_kernel -> float -> transition_kernel
end

(** Mode finding *)
module ModeFinding : sig
  type mode = {
    location : Tensor.t;
    density : float;
    covariance : Tensor.t;
    weight : float;
  }

  type mixing_info = {
    between_mode_transitions : int;
    mode_visits : int array;
    last_mode : int;
    total_steps : int;
  }

  val estimate_local_mode : Tensor.t -> (Tensor.t -> float) -> Tensor.t
  val estimate_mode_covariance : Tensor.t list -> Tensor.t -> Tensor.t
  val identify_modes : Tensor.t list -> (Tensor.t -> float) -> mode list
  val track_mixing : Tensor.t list -> mode list -> mixing_info
end

(** Chain analysis *)
module ChainAnalysis : sig
  type analysis_config = {
    window_size : int;
    min_samples : int;
    convergence_threshold : float;
    stability_threshold : float;
  }

  type chain_metrics = {
    effective_samples : float;
    acceptance_rate : float;
    exploration_score : float;
    stability_score : float;
    mode_coverage : float;
  }

  type chain_state = {
    metrics : chain_metrics;
    modes : ModeFinding.mode list;
    transitions : int;
    stable_windows : int;
  }

  val compute_chain_metrics : Tensor.t list -> analysis_config -> chain_metrics
  val monitor_chain_progress : Tensor.t list -> analysis_config -> chain_state
end

(** Parameter exploration *)
module ParameterExploration : sig
  type region_type =
    | Known
    | Boundary
    | Unknown

  type exploration_stats = {
    visited_regions : (int * int) list;
    boundary_points : Tensor.t list;
    unknown_directions : Tensor.t list;
    exploration_score : float;
  }

  val classify_region : Tensor.t list -> Tensor.t * Tensor.t -> Tensor.t -> region_type
  val find_exploration_direction : Tensor.t list -> Tensor.t * Tensor.t -> Tensor.t -> Tensor.t option
  val explore_region : Tensor.t list -> Tensor.t * Tensor.t -> Tensor.t -> Tensor.t option
  val compute_exploration_score : Tensor.t list -> Tensor.t * Tensor.t -> float
end

(** Chain timing *)
module ChainTiming : sig
  type timing_params = {
    burn_in : int;
    min_adaptation_window : int;
    max_adaptation_window : int;
    stabilization_window : int;
    convergence_check_interval : int;
  }

  type adaptation_phase =
    | BurnIn
    | Adaptation
    | Stationary
    | Converged

  type timing_stats = {
    phase : adaptation_phase;
    current_window : int;
    elapsed_steps : int;
    stable_windows : int;
  }

  val determine_phase : timing_stats -> timing_params -> adaptation_phase
  val adapt_window_size : timing_stats -> timing_params -> int
end

(** Main MCMC *)
module IntrepidMCMC : sig
  type mcmc_config = {
    n_chains : int;
    n_samples : int;
    exploration_ratio : float;
    numerical_params : NumericalStability.numerical_params;
    analysis_config : ChainAnalysis.analysis_config;
    timing_params : ChainTiming.timing_params;
    convergence_criteria : ChainConvergence.convergence_criteria;
  }

  type run_stats = {
    chain_states : ChainAnalysis.chain_state array;
    convergence : bool;
    total_modes : int;
    execution_time : float;
  }

  val create_default_config : int -> mcmc_config
  val run_single_chain : mcmc_config -> Tensor.t -> (Tensor.t -> float) -> (Tensor.t -> float) -> 
    Tensor.t list * ChainTiming.timing_stats
  val run_parallel : mcmc_config -> Tensor.t array -> (Tensor.t -> float) -> (Tensor.t -> float) -> run_stats
end