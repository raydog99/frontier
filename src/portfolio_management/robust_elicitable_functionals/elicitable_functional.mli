open Torch
open Scoring_function

type t = {
  name : string;
  scoring_function : Scoring_function.t;
  dimension : int;
}

val create : string -> Scoring_function.t -> int -> t
val evaluate : t -> Tensor.t -> Tensor.t