type log_level = Debug | Info | Warning | Error

val set_log_level : log_level -> unit
val log : log_level -> string -> unit
val debug : string -> unit
val info : string -> unit
val warning : string -> unit
val error : string -> unit