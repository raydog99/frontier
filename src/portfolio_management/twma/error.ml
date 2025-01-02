type t =
  | ConfigError of string
  | DataError of string
  | NetworkError of string
  | TradingError of string
  | SystemError of string

exception TradingException of t

let to_string = function
  | ConfigError msg -> "Configuration Error: " ^ msg
  | DataError msg -> "Data Error: " ^ msg
  | NetworkError msg -> "Network Error: " ^ msg
  | TradingError msg -> "Trading Error: " ^ msg
  | SystemError msg -> "System Error: " ^ msg

let raise error =
  raise (TradingException error)