type t = {
  mutable price: float;
  mutable quantity: float;
}

val create : float -> float -> t
val clear : t -> float -> float -> unit