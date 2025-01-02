exception InvalidData of string
exception ComputationError of string
exception StrategyError of string

val handle_exn : (unit -> 'a) -> ('a, string) result