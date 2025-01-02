open Torch

val spike_trajectory : Tensor.t -> Tensor.t -> Tensor.t
val spike_spike_overlap : Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t
val spike_bulk_overlap : Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t
val compute_critical_time : float -> float
val compute_spike_phase_transition : float -> float -> float