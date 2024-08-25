type log_level = Debug | Info | Warning | Error

let string_of_log_level = function
  | Debug -> "DEBUG"
  | Info -> "INFO"
  | Warning -> "WARNING"
  | Error -> "ERROR"

let log_file = ref "trading_system.log"
let current_log_level = ref Info

let set_log_file filename =
  log_file := filename

let set_log_level level =
  current_log_level := level

let should_log level =
  match !current_log_level, level with
  | Debug, _ -> true
  | Info, (Info | Warning | Error) -> true
  | Warning, (Warning | Error) -> true
  | Error, Error -> true
  | _ -> false

let log level message =
  if should_log level then
    let oc = open_out_gen [Open_append; Open_creat] 0o644 !log_file in
    let timestamp = Unix.gettimeofday () |> Unix.localtime in
    Printf.fprintf oc "[%04d-%02d-%02d %02d:%02d:%02d] [%s] %s\n"
      (timestamp.tm_year + 1900) (timestamp.tm_mon + 1) timestamp.tm_mday
      timestamp.tm_hour timestamp.tm_min timestamp.tm_sec
      (string_of_log_level level) message;
    close_out oc

let debug msg = log Debug msg
let info msg = log Info msg
let warning msg = log Warning msg
let error msg = log Error msg