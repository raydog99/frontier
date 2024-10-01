type t = {
  mutable price: float;
  mutable volume: int;
}

val create : float -> t
val update_price : t -> float -> float -> unit