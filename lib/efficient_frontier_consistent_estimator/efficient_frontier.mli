open Torch

type t

val create : Tensor.t -> Tensor.t -> t
val estimate_parameters : t -> Estimators.t -> Tensor.t * Tensor.t * Tensor.t
val consistent_estimators : t -> Tensor.t * Tensor.t * Tensor.t
val asymptotic_normality : t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t * Tensor.t * Tensor.t
val quadratic_loss : Tensor.t -> Tensor.t -> Tensor.t
val generate_frontier : t -> Estimators.t -> int -> (float * float) list