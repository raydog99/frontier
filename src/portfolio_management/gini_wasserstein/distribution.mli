open Torch

type t
val create : Tensor.t -> t
val gini_index : t -> float
val wasserstein_distance : t -> t -> float
val get_probs : t -> Tensor.t
val get_mean : t -> float
val get_cumsum : t -> Tensor.t
val l1_distance : t -> t -> float
val shifted_bernoulli : float -> t
val entropy : t -> float
val to_array : t -> float array