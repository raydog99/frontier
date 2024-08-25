type t =
  | ConfigError of string
  | DataError of string
  | NetworkError of string
  | TradingError of string
  | SystemError of string

exception TradingException of t

val to_string : t -> string
val raise : t -> 'a