type level = Debug | Info | Warning | Error

val set_log_level : level -> unit
val log : level -> string -> unit