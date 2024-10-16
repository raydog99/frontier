type t

val create : int -> int -> (float array * float) list -> t
val predict : t -> float array -> float
val feature_importance : t -> float array