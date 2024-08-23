type t

val create : unit -> t
val start : t -> string -> unit
val stop : t -> unit
val print : t -> unit