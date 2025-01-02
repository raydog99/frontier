type t

val create : window_size:int -> tolerance:float -> t
val should_stop : t -> float -> bool