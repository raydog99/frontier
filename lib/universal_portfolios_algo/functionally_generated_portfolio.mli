type t

val create : string -> (Market.t -> float array) -> (Market.t -> float) -> t
val generate_weights : t -> Market.t -> float array
val get_name : t -> string
val get_generating_function : t -> (Market.t -> float)

val constant_weighted : float array -> t
val diversity_weighted : float -> t
val entropy_weighted : t
val equal_weighted : int -> t
val market_weighted : t
val diversity_weighted_generalized : float -> float -> t
val volatility_weighted : float -> t
val adaptive_boltzmann : float -> float -> t
val rank_dependent : float -> float -> t
val cross_entropy : float -> t
val log_barrier : float -> t

val relative_entropy : float array -> float array -> float
val supergradient_vector : t -> Market.t -> float array