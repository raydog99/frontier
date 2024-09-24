open Torch
open Distribution

type model = 
  | RichBiased 
  | PersuasionPolarization 
  | StickyDispersion
  | CustomModel of (DiscreteDistribution.t -> Tensor.t)

type t

val create : DiscreteDistribution.t -> model -> t
val step : t -> float -> t
val theorem1 : t -> float * float
val theorem2 : t -> float * float

val simulate_system : DiscreteDistribution.t -> ODESystem.model -> int -> float -> (float * float * float * ODESystem.t) list
val model_to_string : ODESystem.model -> string