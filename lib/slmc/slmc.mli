open Torch

type preconditioner = {
    matrix: Tensor.t;
    eigenvalues: Tensor.t;
    eigenvectors: Tensor.t;
    blocks: Tensor.t array;
    block_size: int;
    probabilities: float array;
  }

type config = {
    dim: int;
    step_size: float;
    num_blocks: int;
    block_size: int;
    relative_tolerance: float;
    max_iterations: int;
    convergence_threshold: float;
  }

type smoothness_params = {
    m: float;  (* relative strong convexity *)
    m_standard: float;  (* standard strong convexity *)
    big_m: float;  (* relative smoothness *)
    big_m_standard: float;  (* standard smoothness *)
  }

type condition_numbers = {
    kappa: float;  (* standard condition number *)
    kappa_rel: float array;  (* block-wise relative condition numbers *)
    kappa_rel_total: float;  (* total relative condition number *)
  }

type block_statistics = {
    smoothness_violations: int;
    avg_gradient_norm: float;
    avg_step_size: float;
    success_rate: float;
    iterations: int;
  }

type sampling_stats = {
    wasserstein_distances: float list;
    kl_divergences: float list;
    smoothness_violations: int;
    acceptance_rate: float;
    effective_sample_size: float;
  }

type convergence_diagnostics = {
    has_converged: bool;
    error_estimate: float;
    theoretical_bound: float;
    actual_iterations: int;
    theoretical_iterations: int;
  }

val check_pl_inequality : 
    f:(Tensor.t -> Tensor.t) ->
    grad_f:(Tensor.t -> Tensor.t) ->
    x:Tensor.t ->
    x_star:Tensor.t ->
    alpha:float -> bool

val wasserstein_distance : Tensor.t -> Tensor.t -> float
val kl_divergence : Tensor.t -> Tensor.t -> float
val compute_effective_sample_size : Tensor.t list -> float

val eigendecomposition : Tensor.t -> Tensor.t * Tensor.t
val create_blocks : Tensor.t -> int -> Tensor.t array
val compute_block_gram : Tensor.t -> Tensor.t
val project_to_block : Tensor.t -> Tensor.t -> Tensor.t
val compute_condition_number : Tensor.t -> float
val compute_relative_condition : Tensor.t -> smoothness_params -> float

val create : 
    Tensor.t -> 
    int -> 
    smoothness_params -> 
    preconditioner

val update : 
    preconditioner -> 
    block_statistics array -> 
    preconditioner

val sample_block : 
    preconditioner -> 
    block_statistics array -> 
    int * Tensor.t

val compute_optimal_probabilities : 
    preconditioner -> 
    smoothness_params -> 
    float array

val lmc : 
    potential:(Tensor.t -> Tensor.t) ->
    grad_potential:(Tensor.t -> Tensor.t) ->
    config:config ->
    Tensor.t -> Tensor.t * sampling_stats

val plmc :
    potential:(Tensor.t -> Tensor.t) ->
    grad_potential:(Tensor.t -> Tensor.t) ->
    config:config ->
    precond:preconditioner ->
    Tensor.t -> Tensor.t * sampling_stats

val compute_optimal_step_size :
    grad_potential:(Tensor.t -> Tensor.t) ->
    params:smoothness_params ->
    x:Tensor.t -> float

val init : 
    config:config ->
    precond:preconditioner ->
    params:smoothness_params ->
    x0:Tensor.t ->
    block_statistics array

val step :
    potential:(Tensor.t -> Tensor.t) ->
    grad_potential:(Tensor.t -> Tensor.t) ->
    config:config ->
    precond:preconditioner ->
    params:smoothness_params ->
    block_stats:block_statistics array ->
    x:Tensor.t ->
    Tensor.t * block_statistics array * sampling_stats

val run_chain :
    potential:(Tensor.t -> Tensor.t) ->
    grad_potential:(Tensor.t -> Tensor.t) ->
    config:config ->
    precond:preconditioner ->
    params:smoothness_params ->
    x0:Tensor.t ->
    Tensor.t list * convergence_diagnostics

val verify_smoothness_conditions :
    grad_potential:(Tensor.t -> Tensor.t) ->
    x:Tensor.t ->
    y:Tensor.t ->
    params:smoothness_params -> bool

val compute_complexity_bound :
    config:config ->
    params:smoothness_params ->
    epsilon:float -> int

val estimate_mixing_time :
    config:config ->
    params:smoothness_params ->
    initial_dist:float ->
    target_error:float -> int

val verify_convergence_rate :
    samples:Tensor.t list ->
    params:smoothness_params ->
    float * bool

val adapt_step_size :
    current:float ->
    success_rate:float ->
    smoothness:float -> float

val adapt_block_size :
    config:config ->
    stats:block_statistics array ->
    condition_number:float -> int

val adapt_probabilities :
    precond:preconditioner ->
    stats:block_statistics array ->
    float array

val update_block_statistics :
    stats:block_statistics array ->
    block_idx:int ->
    success:bool ->
    grad_norm:float ->
    step_size:float ->
    block_statistics array

val check_convergence :
    samples:Tensor.t list ->
    config:config ->
    params:smoothness_params ->
    convergence_diagnostics

val compute_error_estimate :
    samples:Tensor.t list ->
    target:Tensor.t option ->
    float

val monitor_chain :
    samples:Tensor.t list ->
    stats:sampling_stats ->
    unit

val generate_diagnostics_report :
    samples:Tensor.t list ->
    stats:sampling_stats ->
    diagnostics:convergence_diagnostics ->
    string