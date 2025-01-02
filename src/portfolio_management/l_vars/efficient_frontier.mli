open Torch
open Time_dependent_strategy

type t

val create : float -> float -> int -> float -> float -> float -> float -> float -> ?parameter_shift:ParameterShift.t option -> unit -> t
val calculate_frontier : t -> int -> (Tensor.t * Tensor.t)
val optimize_utility : t -> Utility.t -> Tensor.t
val calculate_var : t -> float -> (Tensor.t * float)
val optimize_time_dependent : t -> Time_dependent_strategy.t -> Tensor.t
val analyze_strategy : t -> Tensor.t -> unit
val optimize_almgren_chriss : t -> Tensor.t
val optimize_dynamic : t -> (Tensor.t -> Tensor.t) -> Tensor.t
val optimize_constrained : t -> (Tensor.t -> Tensor.t) -> Tensor.t