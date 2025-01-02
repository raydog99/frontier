open Torch

type t = {
  dim: int;
  weights: Tensor.t;
  intervention_weights: Tensor.t;
  noise_mean: Tensor.t;
  max_weight: float;
  max_noise: float;
}

val create : ?max_weight:float -> ?max_noise:float -> int -> t
val simulate : t -> Graph.NodeSet.t -> Tensor.t -> Tensor.t
val estimate_means : (Tensor.t * Graph.NodeSet.t) list -> Graph.NodeSet.t -> 
  Tensor.t * Tensor.t Graph.NodeMap.t
val check_weight_constraint : t -> bool
val check_intervention_regularity : t -> float -> 
  (Tensor.t * Graph.NodeSet.t) list -> bool