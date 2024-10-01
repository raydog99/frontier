type t = {
  mutable interest_rate: float;
  mutable money_supply: float;
}

val create : float -> float -> t
val set_interest_rate : t -> float -> float -> unit
val adjust_money_supply : t -> float -> float -> unit