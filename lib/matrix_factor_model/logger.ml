type level = Debug | Info | Warning | Error

let current_level = ref Info

let set_log_level level =
  current_level := level

let level_to_string = function
  | Debug -> "DEBUG"
  | Info -> "INFO"
  | Warning -> "WARNING"
  | Error -> "ERROR"

let log level message =
  if level >= !current_level then
    Printf.printf "[%s] [%s] %s\n"
      (Unix.gettimeofday () |> int_of_float |> string_of_int)
      (level_to_string level)
      message