open Torch

type convergence_stats = {
  potential_scale_reduction: float;
  effective_sample_size: float;
  autocorrelation_time: float;
  spectral_gap_estimate: float;
}

(** Gelman-Rubin diagnostic for multiple chains *)
val gelman_rubin : Tensor.t list -> int -> float

val effective_sample_size : Tensor.t -> int -> float

val analyze_convergence : Tensor.t list -> int -> convergence_stats

(** Check if chains have converged sufficiently *)
val has_converged : convergence_stats -> float -> bool