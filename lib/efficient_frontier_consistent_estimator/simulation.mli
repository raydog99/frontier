open Efficient_frontier

type scenario =
  | Normal
  | StudentT of float
  | CCCGARCH

val generate_data : int -> int -> scenario -> Efficient_frontier.t
val run_simulation : int -> int -> scenario -> int -> (string * float * (float * float) * float * (float * float) * float * (float * float) * float * float * float) list
val compare_estimators : (string * float * (float * float) * float * (float * float) * float * (float * float) * float * float * float) list -> (string * float * float * float) list