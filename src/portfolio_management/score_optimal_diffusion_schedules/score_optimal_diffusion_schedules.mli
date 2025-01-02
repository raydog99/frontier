open Torch

  type schedule = {
    drift : float -> float;      (** f(t) *)
    diffusion : float -> float;  (** g(t) *)
    mean_scale : float -> float; (** s(t) *)
    var_scale : float -> float;  (** σ²(t) *)
  }

  type discretization = float array

val forward_step : Tensor.t -> float -> schedule -> float -> Tensor.t -> Tensor.t
val backward_step : Tensor.t -> float -> schedule -> (Tensor.t -> float -> Tensor.t) -> float -> Tensor.t -> Tensor.t
val create_vp_schedule : float -> float -> schedule

val stein_divergence : (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t
val local_cost : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> Tensor.t

val predict : Tensor.t -> float -> float -> schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t
val correct : Tensor.t -> float -> schedule -> (Tensor.t -> float -> Tensor.t) -> float -> float -> Tensor.t

val corrector_cost : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> float -> Tensor.t
val predictor_cost : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> float -> Tensor.t

val estimate_trace : (Tensor.t -> Tensor.t) -> Tensor.t -> int -> Tensor.t * Tensor.t
val adaptive_trace_estimation : (Tensor.t -> Tensor.t) -> Tensor.t -> float -> int -> Tensor.t

val log_det_lu : Tensor.t -> Tensor.t
val log_det_stochastic : Tensor.t -> int -> Tensor.t

val calculate_snr : schedule -> float -> float
val generate_snr_schedule : schedule -> int -> float array
val quality_metric : (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> Tensor.t
val optimize_quality : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float array -> int -> float array
val pathwise_kl : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> float -> Tensor.t
val optimize_pathwise_kl : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float array -> float array
val fisher_metric : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> Tensor.t
val generate_fisher_schedule : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> int -> float array

val antithetic_sampling : (Tensor.t -> Tensor.t) -> Tensor.t -> int -> Tensor.t
val control_variate_sampling : (Tensor.t -> Tensor.t) -> Tensor.t -> (Tensor.t -> Tensor.t) -> int -> Tensor.t
val importance_sampling : (Tensor.t -> Tensor.t) -> Tensor.t -> (unit -> Tensor.t) -> int -> Tensor.t

module GeometricOptimization : sig
  module PathGenerator : sig
    type t = {
      generator : float -> float;
      derivative : float -> float;
      metric : float -> float;
    }

    val create : (float -> float) -> t
  end

    val energy_density : PathGenerator.t -> schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> Tensor.t
    val total_energy : PathGenerator.t -> schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> int -> Tensor.t

    val metric_tensor : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> Tensor.t
    val christoffel_symbols : Tensor.t -> Tensor.t
    val integrate_geodesic : (Tensor.t -> float -> Tensor.t) -> Tensor.t -> Tensor.t -> schedule -> int -> Tensor.t array

    val compute : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> int -> float array
    val verify_optimality : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float array -> bool
end

module ComparisonFramework : sig
  type approach_result = {
    schedule : float array;
    quality : float;
    computation_time : float;
    memory_usage : int;
  }

  val compare_approaches : schedule -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float array ->
    (string * approach_result) list
end

module LinearSchedule : sig
    val create : float -> float -> schedule
end

module CosineSchedule : sig
    val create : ?epsilon:float -> unit -> schedule
end

module AdaptiveSchedule : sig
  type adaptive_state

  val init : schedule -> int -> adaptive_state
  val update_noise_levels : adaptive_state -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> float -> Tensor.t -> schedule
  val adapt_schedule : adaptive_state -> (Tensor.t -> float -> Tensor.t) -> Tensor.t -> int -> schedule
end