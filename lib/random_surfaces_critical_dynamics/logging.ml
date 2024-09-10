type log_level = Debug | Info | Warning | Error

let string_of_log_level = function
  | Debug -> "DEBUG"
  | Info -> "INFO"
  | Warning -> "WARNING"
  | Error -> "ERROR"

let current_log_level = ref Info

let set_log_level level =
  current_log_level := level

let log level message =
  if level >= !current_log_level then
    Printf.eprintf "[%s] %s: %s\n"
      (string_of_log_level level)
      (Sys.time () |> int_of_float |> string_of_int)
      message

let debug msg = log Debug msg
let info msg = log Info msg
let warning msg = log Warning msg
let error msg = log Error msg

let with_error_logging f x =
  try f x with
  | e ->
      error (Printf.sprintf "Exception caught: %s" (Printexc.to_string e));
      raise e