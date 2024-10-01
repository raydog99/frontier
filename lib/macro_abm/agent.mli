type t = {
  id: int;
  mutable wealth: float;
}

val create : int -> float -> t
val update_wealth : t -> float -> unit