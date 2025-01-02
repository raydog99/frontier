open Printf

type log_level = DEBUG | INFO | WARNING | ERROR

let current_log_level = ref INFO

let set_log_level level = current_log_level := level

let log level message =
  if level >= !current_log_level then
    match level with
    | DEBUG -> printf "[DEBUG] %s\n" message
    | INFO -> printf "[INFO] %s\n" message
    | WARNING -> printf "[WARNING] %s\n" message
    | ERROR -> printf "[ERROR] %s\n" message

let debug msg = log DEBUG msg
let info msg = log INFO msg
let warning msg = log WARNING msg
let error msg = log ERROR msg

exception PolyModelError of string

let raise_error msg =
  error msg;
  raise (PolyModelError msg)