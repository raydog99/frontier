open Torch

type t = Tensor.t -> Tensor.t -> Tensor.t

val squared_error : t
val pinball_loss : float -> t
val expectile_loss : float -> t
val var_score : float -> t
val es_score : float -> t
val b_homogeneous_mean : float -> t