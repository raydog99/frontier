type t

val create : int -> float array -> float array array -> t
val update : t -> float array -> t
val get_weight : t -> int -> float
val get_weights : t -> float array
val size : t -> int
val get_volatility : t -> float array array
val normalize : float array -> float array
val rank_based_diffusion : float array -> float -> float -> float -> float -> int -> t list
val diversity_coefficient : t -> float -> float
val entropy : t -> float