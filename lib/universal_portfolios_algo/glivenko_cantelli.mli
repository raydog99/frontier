val empirical_measure : Portfolio.t list -> Market.t list -> int -> float list
val true_measure : Portfolio.t list -> Market.t -> float list
val supremum_norm : float list -> float list -> float
val verify_gc_property : Portfolio.t list -> Market.t list -> int -> float -> bool
val vc_dimension : Portfolio.t list -> int
val vapnik_chervonenkis_bound : Portfolio.t list -> Market.t list -> int -> float -> bool
val rademacher_complexity : Portfolio.t list -> Market.t list -> int -> float
val rademacher_bound : Portfolio.t list -> Market.t list -> int -> float -> bool