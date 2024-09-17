open Torch

type t = {
    features : Tensor.t;
    returns : Tensor.t;
  }

val load_data : string -> (t, string) result
val preprocess : t -> int -> (t, string) result
val split : t -> (t * t * t, string) result
val get_batch : t -> int -> (t, string) result
val parallel_preprocess : t -> int -> int -> t
val bootstrap : t -> t