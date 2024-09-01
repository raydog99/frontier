open Torch

val evt_tail_risk : Tensor.t -> float -> float

module Copula : sig
  type t = 
    | Gaussian of Tensor.t
    | Student of { nu: float; corr: Tensor.t }

  val gaussian_copula : Tensor.t -> t
  val student_copula : Tensor.t -> float -> t
  val sample : t -> int -> Tensor.t
end

val compute_cvar : Tensor.t -> float -> float
val stress_test : Tensor.t -> Tensor.t -> Tensor.t list -> float list