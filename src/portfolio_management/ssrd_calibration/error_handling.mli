exception CalibrationError of string

val handle_error : (unit -> 'a) -> 'a