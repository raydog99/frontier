type t

val create : unit -> t
val update : t -> Params.t -> float -> [`Model1 | `Model2] -> unit
val initialize : t -> Params.t -> [`Model1 | `Model2] -> unit
val get_h2 : t -> float
val get_h1 : t -> float
val get_h0 : t -> float