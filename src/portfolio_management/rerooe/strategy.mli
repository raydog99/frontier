type t =
  | TWAP
  | VWAP of { volume_curve: float -> float }
  | Optimal

val execute : t -> Model.t -> Params.t -> Value_function.t -> float -> [`Model1 | `Model2] -> float