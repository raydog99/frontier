open Types

exception InvalidInputData of string

val read_market_data : string -> market_index array
val save_results : analysis_summary -> string -> unit