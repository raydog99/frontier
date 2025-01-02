open Types

val read_market_data : string -> string -> market_data
val calculate_errors : ('a * float) list -> ('a * float) list -> market_data -> float * float