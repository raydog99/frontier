type reference_point = Initial | Strike

type option_type = Call | Put

type t

exception InvalidParameterError of string

val create : float -> float -> float -> float -> float -> float -> float -> float -> float -> float -> int -> int -> reference_point -> t

val simulate_paths_parallel : t -> float array array

val payoff : option_type -> float -> float -> float

val longstaff_schwartz_optimized : t -> option_type -> float

val pdifmp_price : t -> option_type -> float

val ls_pdifmp_price : t -> option_type -> float

val compare_methods : t -> option_type -> float * float * float

val run_all_experiments : unit -> unit

val run_unit_tests : unit -> unit