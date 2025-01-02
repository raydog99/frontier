open Types

type log_level = Debug | Info | Warning | Error

let current_log_level = ref Info

let set_log_level level = current_log_level := level

let log level message =
  if level >= !current_log_level then
    Printf.fprintf stderr "[%s] %s\n"
      (match level with
       | Debug -> "DEBUG"
       | Info -> "INFO"
       | Warning -> "WARNING"
       | Error -> "ERROR")
      message

let debug msg = log Debug msg
let info msg = log Info msg
let warning msg = log Warning msg
let error msg = log Error msg