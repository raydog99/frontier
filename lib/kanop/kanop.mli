open Types

type t

val create : option_params -> kan_params -> option_type -> option_style -> int -> int -> t
val train : t -> int -> float -> unit
val price : t -> float
val delta : t -> float
val gamma : t -> float
val theta : t -> float