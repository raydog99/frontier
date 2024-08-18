type log_level = DEBUG | INFO | WARNING | ERROR

val set_log_level : log_level -> unit
val debug : string -> unit
val info : string -> unit
val warning : string -> unit
val error : string -> unit

exception PolyModelError of string

val raise_error : string -> 'a