open Torch

val create_bernoulli_matrix : int -> float -> Tensor.t
val bernoulli_bulk_overlap : float -> float -> Tensor.t -> Tensor.t -> Tensor.t
val bernoulli_spike_overlap : int -> float -> float -> Tensor.t