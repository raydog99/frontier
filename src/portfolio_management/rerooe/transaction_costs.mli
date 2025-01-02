type t =
  | Fixed of float
  | Linear of float
  | Nonlinear of (float -> float)

val calculate_cost : t -> float -> float -> float
val apply_cost : t -> float -> float -> float