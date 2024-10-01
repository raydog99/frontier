open Sector

type t = {
  id: int;
  mutable income: float;
  mutable consumption: float;
  mutable labor_supply: float;
  mutable reservation_wage: float;
  mutable housing: float;
  mutable mortgage: float;
  mutable financial_assets: float;
}

val create : int -> float -> float -> t
val consume : t -> float -> Sector.t list -> (Sector.t * float) list
val set_labor_supply : t -> float -> unit
val update_reservation_wage : t -> float -> unit
val buy_house : t -> float -> float -> unit
val sell_house : t -> float -> unit
val pay_mortgage : t -> float -> unit
val decide_consumption : t -> float -> unit
val decide_labor_supply : t -> float -> unit
val invest : t -> float -> unit