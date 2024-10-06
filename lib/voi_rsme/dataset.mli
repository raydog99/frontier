open Torch

type t

val create : Tensor.t -> string array -> t
val split_train_test : t -> int -> Tensor.t * Tensor.t
val create_predictors : t -> int -> int -> Tensor.t
val create_response : t -> int -> Tensor.t
val create_rolling_windows : t -> int -> int -> (Tensor.t * Tensor.t) list