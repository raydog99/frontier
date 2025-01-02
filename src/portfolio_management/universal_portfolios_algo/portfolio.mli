type t

val create : int -> float array -> t
val update : t -> float array -> t
val get_weight : t -> int -> float
val get_weights : t -> float array
val size : t -> int
val relative_value : t -> Market.t list -> int -> float
val log_relative_value : t -> Market.t list -> int -> float
val normalize : t -> t