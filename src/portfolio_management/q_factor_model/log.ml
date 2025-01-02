type level = Debug | Info | Warning | Error

let current_level = ref Info

let set_level level =
  current_level := level

let log level msg =
  if level >= !current_level then
    Printf.eprintf "[%s] %s\n"
      (match level with
       | Debug -> "DEBUG"
       | Info -> "INFO"
       | Warning -> "WARNING"
       | Error -> "ERROR")
      msg