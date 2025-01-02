type log_level = Debug | Info | Warning | Error

let current_log_level = ref Info

let set_log_level level = current_log_level := level

let log level message =
  let level_str = match level with
    | Debug -> "DEBUG"
    | Info -> "INFO"
    | Warning -> "WARNING"
    | Error -> "ERROR"
  in
  if level >= !current_log_level then
    Printf.printf "[%s] %s\n" level_str message

let debug msg = log Debug msg
let info msg = log Info msg
let warning msg = log Warning msg
let error msg = log Error msg