type log_level = Debug | Info | Warning | Error

let current_log_level = ref Info

let set_log_level level = current_log_level := level

let log level msg =
  if level >= !current_log_level then
    match level with
    | Debug -> Printf.eprintf "[DEBUG] %s\n" msg
    | Info -> Printf.printf "[INFO] %s\n" msg
    | Warning -> Printf.eprintf "[WARNING] %s\n" msg
    | Error -> Printf.eprintf "[ERROR] %s\n" msg

let debug msg = log Debug msg
let info msg = log Info msg
let warning msg = log Warning msg
let error msg = log Error msg