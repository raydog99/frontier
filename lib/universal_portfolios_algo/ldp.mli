type rate_function = Portfolio.t -> float

val calculate_rate_function : Portfolio.t list -> Market.t list -> int -> rate_function
val verify_ldp : WealthDistribution.t -> rate_function -> float -> bool
val asymptotic_equipartition : Portfolio.t list -> Market.t list -> int -> float -> bool
val varadhan_laplace_principle : (Portfolio.t -> float) -> Portfolio.t list -> Market.t list -> int -> float
val large_deviation_upper_bound : Portfolio.t list -> rate_function -> float
val large_deviation_lower_bound : Portfolio.t list -> rate_function -> float
val gartner_ellis_theorem : (float -> int -> float) -> int -> float -> int -> float option
val scaled_cumulant_generating_function : Portfolio.t list -> Market.t list -> int -> float -> float
val rate_function_from_scgf : (float -> int -> float) -> int -> float -> float
val verify_ldp_with_gartner_ellis : Portfolio.t list -> Market.t list -> int -> float -> bool