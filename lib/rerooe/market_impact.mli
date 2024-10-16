type t =
  | Linear of { permanent: float; temporary: float }
  | SquareRoot of { permanent: float; temporary: float }
  | Exponential of { permanent: float; temporary: float; decay: float }

val calculate_impact : t -> float -> float -> float * float
val apply_impact : t -> float -> float -> float