open Types
open Config

val admlsa : t -> framework -> loss_function -> float -> int -> float Lwt.t