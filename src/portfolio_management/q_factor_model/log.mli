type level = Debug | Info | Warning | Error

val set_level : level -> unit
val log : level -> string -> unit