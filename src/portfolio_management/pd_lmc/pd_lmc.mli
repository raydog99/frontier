open Torch

(** Core types *)
type distribution = {
  samples: Tensor.t;
  log_prob: Tensor.t -> float;
}

type constraint_fn = {
  g: Tensor.t -> Tensor.t;  (** Inequality constraints *)
  h: Tensor.t -> Tensor.t;  (** Equality constraints *)
  grad_g: Tensor.t -> Tensor.t;
  grad_h: Tensor.t -> Tensor.t;
}

type algorithm_params = {
  step_size: float;
  num_iterations: int;
  batch_size: int;
  device: Device.t;
}

(** Core utility functions *)
val kl_divergence : distribution -> distribution -> Tensor.t -> float
val wasserstein_gradient : (Tensor.t -> float) -> Tensor.t -> Tensor.t
val compute_potential_energy : (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> 
                             (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t

(** Wasserstein geometry module *)
module Wasserstein : sig
  val gradient_flow : f:(Tensor.t -> Tensor.t) -> mu:Tensor.t -> step_size:float -> Tensor.t
  val wasserstein2_distance : mu:Tensor.t -> nu:Tensor.t -> Tensor.t
  val wasserstein2_gradient : kl_div:(Tensor.t -> Tensor.t) -> mu:Tensor.t -> Tensor.t
  val sinkhorn_algorithm : source:Tensor.t -> target:Tensor.t -> epsilon:float -> max_iter:int -> Tensor.t * Tensor.t
end

(** Fokker-Planck dynamics *)
module FokkerPlanck : sig
  type state = {
    position: Tensor.t;
    velocity: Tensor.t;
    time: float;
  }

  val evolve_distribution : initial_state:state -> drift:(Tensor.t -> Tensor.t) -> 
                          diffusion:(Tensor.t -> Tensor.t) -> dt:float -> num_steps:int -> state list
end

(** Langevin dynamics *)
module Langevin : sig
  val langevin_step : Tensor.t -> (Tensor.t -> Tensor.t) -> float -> Tensor.t
  val primal_step : Tensor.t -> (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> 
                   (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> algorithm_params -> Tensor.t
  val dual_step : Tensor.t -> Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t) -> 
                 (Tensor.t -> Tensor.t) -> algorithm_params -> Tensor.t * Tensor.t
end

(** Numerical stability enhancements *)
module NumericalEnhanced : sig
  type stability_config = {
    regularization_strength: float;
    condition_threshold: float;
    scaling_factor: float;
    precision_mode: [ `Single | `Double | `Mixed ];
  }

  val stabilize_matrix : matrix:Tensor.t -> config:stability_config -> Tensor.t
  val stable_gradient_computation : f:(Tensor.t -> Tensor.t) -> x:Tensor.t -> config:stability_config -> Tensor.t
  val condition_number : matrix:Tensor.t -> float
  val stable_inverse : matrix:Tensor.t -> config:stability_config -> Tensor.t
end

(** Core PDLMC algorithm *)
module PDLMC : sig
  val run : target:distribution -> constraints:constraint_fn -> params:algorithm_params -> 
           init_x:Tensor.t -> init_lambda:Tensor.t -> init_nu:Tensor.t -> Tensor.t
end

(** Time-scale separation *)
module TimeScales : sig
  type time_scale_config = {
    fast_step_size: float;
    slow_step_size: float;
    scale_adaptation_freq: int;
    scale_adaptation_rate: float;
  }

  type scale = {
    primal_step: float;
    dual_step: float;
    ratio: float;
  }

  val adaptive_time_scales : primal_grad:Tensor.t -> dual_grad:Tensor.t -> current_scale:scale -> scale
  val multi_scale_integration : config:time_scale_config -> fast_system:Tensor.t -> 
                              slow_system:Tensor.t -> coupling:(Tensor.t -> Tensor.t) -> Tensor.t * Tensor.t
end

(** Parallel processing *)
module ParallelEnhanced : sig
  type parallel_config = {
    num_chains: int;
    sync_frequency: int;
    temperature_ladder: float array;
    communication_type: [ `AllReduce | `Ring | `Hierarchical ];
  }

  val parallel_tempering : config:parallel_config -> log_prob:(Tensor.t -> float) -> 
                         init_states:Tensor.t array -> Tensor.t list
end

(** Advanced sampling methods *)
module SamplingEnhanced : sig
  type sampling_scheme = MALA | HMC | NUTS

  val sample : scheme:sampling_scheme -> log_prob:(Tensor.t -> float) -> init_state:Tensor.t -> 
              num_samples:int -> step_size:float -> Tensor.t list
end

(** Convergence analysis *)
module ConvergenceAnalysis : sig
  type convergence_metric = {
    iteration: int;
    kl_div: float;
    wasserstein_dist: float;
    constraint_violation: float;
    dual_gap: float;
    grad_norm: float;
  }

  val compute_metrics : state:'a -> target:distribution -> constraints:constraint_fn list -> 
                       iteration:int -> convergence_metric
  val verify_convergence : metrics:convergence_metric list -> tolerance:float -> bool
  val analyze_local_convergence : trajectory:Tensor.t list -> target:distribution -> 
                                constraints:constraint_fn list -> params:algorithm_params -> float * float array
end

(** Error propagation *)
module ErrorPropagation : sig
  type error_bounds = {
    gradient_error: Tensor.t;
    constraint_error: Tensor.t;
    stability_constant: float;
  }

  val analyze_error_propagation : state:Tensor.t -> update_fn:(Tensor.t -> Tensor.t) -> 
                                num_steps:int -> error_bounds
end