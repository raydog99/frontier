open Torch

type mgarch_model =
  | DCC
  | BEKK

val estimate_dcc_garch : Tensor.t -> int -> int -> float -> ((float * float * float * float) array * (float * float))
val forecast_dcc_garch : Tensor.t -> (float * float * float * float) array -> (float * float) -> int -> (Tensor.t array * Tensor.t)