type t = {
  id: int;
  mutable reserves: float;
  mutable loans: float;
  mutable interest_rate: float;
  mutable equity: float;
}

val create : int -> float -> float -> t
val set_interest_rate : t -> float -> unit
val calculate_loan_rate : t -> float -> float -> float
val process_loan : t -> float -> float -> float option
val receive_payment : t -> float -> unit
val update_equity : t -> float -> unit