open Sector

type t = {
  id: int;
  sector: Sector.t;
  mutable capital: float;
  mutable labor: float;
  mutable intermediate_inputs: (Sector.t * float) list;
  mutable production: float;
  mutable price: float;
  mutable inventory: float;
  mutable expected_demand: float;
  mutable target_production: float;
  mutable debt: float;
  mutable equity: float;
}

val create : int -> Sector.t -> float -> float -> t
val produce : t -> (Sector.t * (Sector.t * float) list) list -> unit
val set_price : t -> float -> float -> float -> unit
val update_expected_demand : t -> float -> float -> float -> unit
val set_target_production : t -> float -> float -> float -> float -> unit
val invest : t -> float -> float
val request_loan : t -> float -> float -> unit
val repay_loan : t -> float -> unit
val update_equity : t -> float -> unit