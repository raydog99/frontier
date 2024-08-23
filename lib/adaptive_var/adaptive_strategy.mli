type t

val create : theta:float -> r:float -> ca:float -> m:int -> t
val update : t -> performance:float -> convergence_rate:float -> error:float -> level:int -> unit
val get_params : t -> float * float * float * int