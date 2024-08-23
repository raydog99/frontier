open Types

type t = framework

val epsilon : t -> (float -> float)
val optimal_theta : t -> float
val optimal_complexity : t -> (float -> float)