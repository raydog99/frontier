open Torch

module type Proposal = sig
  type t = {
    mu: Tensor.t;      (* Location parameter *)
    sigma: Tensor.t;    (* Scale parameter *)
    dim: int;          (* Dimensionality *)
    log_det_sigma: float; (* Log determinant of sigma *)
  }

  val create : mu:Tensor.t -> sigma:Tensor.t -> dim:int -> t
  val log_prob : t -> Tensor.t -> Tensor.t
  val sample : t -> n_samples:int -> Tensor.t
end

val log_sum_exp : x:Tensor.t -> dim:int -> Tensor.t
val compute_covariance : samples:Tensor.t -> weights:Tensor.t -> Tensor.t
val safe_log : Tensor.t -> Tensor.t
val stable_softmax : Tensor.t -> Tensor.t
val stable_importance_weights : Tensor.t -> Tensor.t

module type WeightingScheme = sig
  type t = Standard | DeterministicMixture
  
  val compute_dm_weights : 
    samples:Tensor.t -> 
    proposals:Proposal.t array -> 
    target_log_prob:(Tensor.t -> Tensor.t) -> 
    Tensor.t
end

module type Monitor = sig
  type statistics = {
    mean_estimate: Tensor.t;
    covariance_estimate: Tensor.t;
    ess: float;
    log_weights: Tensor.t;
    kl_estimate: float option;
    mode_coverage: float;
  }

  val compute_ess : weights:Tensor.t -> float
  val compute_statistics : 
    samples:Tensor.t -> 
    weights:Tensor.t -> 
    true_mean_opt:Tensor.t option -> 
    statistics
end

module type ModeTracking = sig
  type mode = {
    location: Tensor.t;
    covariance: Tensor.t;
    weight: float;
    visits: int;
  }

  type mode_state = {
    discovered_modes: mode list;
    mode_distances: float array array option;
    min_mode_distance: float;
  }

  val create_mode_state : dim:int -> mode_state
  val update_mode_tracking : 
    state:mode_state -> 
    samples:Tensor.t -> 
    weights:Tensor.t -> 
    target_log_prob:(Tensor.t -> Tensor.t) -> 
    mode_state

  val compute_mode_distance : mode -> mode -> float
  val is_new_mode : mode -> mode list -> min_distance:float -> bool
end

module type HMC = sig
  type config = {
    n_leapfrog: int;
    step_size: float;
    mass: Tensor.t;
  }

  val create_config : ?n_leapfrog:int -> ?step_size:float -> dim:int -> config
  val integrate : 
    target_log_prob:(Tensor.t -> Tensor.t) -> 
    initial_pos:Tensor.t -> 
    mass:Tensor.t -> 
    epsilon:float -> 
    n_steps:int -> 
    Tensor.t * float

  val sample : 
    position:Tensor.t -> 
    config:config -> 
    target_log_prob:(Tensor.t -> Tensor.t) -> 
    Tensor.t * bool
end

module type Adaptation = sig
  type t = {
    weighted_locations: Tensor.t array;
    hmc_locations: Tensor.t array;
  }

  val generate_weighted_locations : 
    samples:Tensor.t -> 
    weights:Tensor.t -> 
    n_proposals:int -> 
    n_samples:int -> 
    Tensor.t array

  val generate_hmc_locations : 
    proposals:Proposal.t array -> 
    config:HMC.config -> 
    target_log_prob:(Tensor.t -> Tensor.t) -> 
    Tensor.t array
end

module type CurvedAdaptation = sig
  type metric_type = 
    | Euclidean 
    | Riemannian of {epsilon: float; decay: float}

  val compute_riemannian_metric : 
    pos:Tensor.t -> 
    grad:Tensor.t -> 
    epsilon:float -> 
    Tensor.t

  val curved_leapfrog_step :
    pos:Tensor.t ->
    mom:Tensor.t ->
    target_log_prob:(Tensor.t -> Tensor.t) ->
    mass:Tensor.t ->
    epsilon:float ->
    metric_type:metric_type ->
    Tensor.t * Tensor.t

  val generate_preliminary_locations :
    samples:Tensor.t ->
    weights:Tensor.t ->
    config:HPMC.config ->
    Tensor.t array
end

module type AdaptiveScheduler = sig
  type phase = {
    exploration_weight: float;
    refinement_weight: float;
    step_size: float;
    n_leapfrog: int;
  }

  type schedule = {
    phases: phase array;
    current_phase: int;
    phase_length: int;
    total_iterations: int;
  }

  val create_schedule : 
    n_iterations:int -> 
    initial_step_size:float -> 
    n_leapfrog:int -> 
    schedule

  val get_current_phase : schedule -> int -> phase
end

module type IntegratedAdaptation = sig
  type state = {
    phase: [`Exploration | `Refinement | `Mixed];
    step_size: float;
    n_leapfrog: int;
    metric: CurvedAdaptation.metric_type;
    mode_state: ModeTracking.mode_state;
    adaptation_weights: float array;
    accepted_moves: int;
  }

  val create_state : HPMC.config -> state
  
  val adapt_step :
    state:state ->
    config:HPMC.config ->
    samples:Tensor.t ->
    weights:Tensor.t ->
    target_log_prob:(Tensor.t -> Tensor.t) ->
    state * Tensor.t array
end

module type HPMC = sig
  type config = {
    n_proposals: int;
    n_samples: int;
    n_iterations: int;
    dim: int;
    step_size: float;
    n_leapfrog: int;
  }

  type state = {
    samples: Tensor.t;
    weights: Tensor.t;
    proposals: Proposal.t array;
    integrated_adaptation: IntegratedAdaptation.state;
    iteration: int;
    mode_tracking: ModeTracking.mode_state;
    stats: Monitor.statistics;
  }

  val create_config : 
    n_proposals:int -> 
    n_samples:int -> 
    n_iterations:int -> 
    dim:int -> 
    config

  val init_state : config -> state
  
  val run : 
    config -> 
    target_log_prob:(Tensor.t -> Tensor.t) -> 
    state
end

module type Benchmark = sig
  type distribution_type = 
    | MultiModal2D
    | HighDimBimodal of int
    | BananaShaped of {dim: int; b: float}
    
  type result = {
    ess_history: float array;
    mode_discovery: int array;
    mean_error: float array;
    runtime: float;
    target_evals: int;
    acceptance_rates: float array;
  }

  val create_target : 
    distribution_type -> 
    (Tensor.t -> Tensor.t) * Tensor.t array

  val run_benchmark : 
    ?n_repeats:int -> 
    distribution_type -> 
    HPMC.config -> 
    result array
end

module type Diagnostics = sig
  type diagnostic_report = {
    ess_trend: float array;
    mode_stability: float;
    weight_entropy: float;
    grad_norm_stats: float * float;
    proposal_spread: float;
  }

  val compute_weight_entropy : Tensor.t -> float
  val compute_proposal_spread : Proposal.t array -> float
  val generate_report : HPMC.state -> diagnostic_report
end