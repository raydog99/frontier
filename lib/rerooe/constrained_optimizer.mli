type constraint_type =
  | MinWeight of float
  | MaxWeight of float
  | SectorExposure of string * float * float

type t

val create : Portfolio_optimizer.optimization_method -> constraint_type list -> t
val optimize : Portfolio.t -> t -> float array