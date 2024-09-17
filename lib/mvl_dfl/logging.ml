type log_level = Debug | Info | Warning | Error

type log_entry = {
  timestamp: float;
  level: log_level;
  message: string;
}

let log_buffer = Queue.create ()

let string_of_log_level = function
  | Debug -> "DEBUG"
  | Info -> "INFO"
  | Warning -> "WARNING"
  | Error -> "ERROR"

let current_log_level = ref Info

let set_log_level level = current_log_level := level

let log level message =
  if level >= !current_log_level then begin
    let entry = {
      timestamp = Unix.gettimeofday ();
      level;
      message;
    } in
    Queue.add entry log_buffer;
    Printf.printf "[%s] %s: %s\n" 
      (Unix.localtime entry.timestamp |> Printf.sprintf "%04d-%02d-%02d %02d:%02d:%02d" 
        (1900 + entry.timestamp.tm_year) (entry.timestamp.tm_mon + 1) entry.timestamp.tm_mday 
        entry.timestamp.tm_hour entry.timestamp.tm_min entry.timestamp.tm_sec)
      (string_of_log_level level) 
      message
  end

let debug message = log Debug message
let info message = log Info message
let warning message = log Warning message
let error message = log Error message

let save_log_to_file filename =
  let oc = open_out filename in
  Queue.iter (fun entry ->
    Printf.fprintf oc "[%f] [%s] %s\n" 
      entry.timestamp 
      (string_of_log_level entry.level) 
      entry.message
  ) log_buffer;
  close_out oc