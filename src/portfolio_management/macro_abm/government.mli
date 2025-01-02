type t = {
  mutable tax_rate: float;
  mutable spending: float;
  mutable debt: float;
}

val create : float -> t
val collect_taxes : t -> float -> float
val adjust_spending : t -> float -> float -> unit
val issue_debt : t -> unit