open Torch

val limiting_rescaled_mean_squared_overlap : float -> float -> Tensor.t -> Tensor.t -> Tensor.t
val find_optimal_overlap : float -> float -> Tensor.t -> Tensor.t
val check_interlacing : float -> float -> Tensor.t -> Tensor.t -> Tensor.t
val compute_overlap_distribution : int -> float -> float -> Tensor.t -> Tensor.t
val compute_bulk_edge : float -> float -> float