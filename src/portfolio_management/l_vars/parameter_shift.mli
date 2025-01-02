open Torch

type scenario = {
  probability: float;
  volatility_multiplier: float;
  drift: float;
  permanent_impact_multiplier: float;
  temporary_impact_multiplier: float;
}

type t

val create : float -> scenario list -> t
val apply : t -> Tensor.t -> Tensor.t
val get_expected_parameters : t -> float -> float -> float -> (float * float * float)