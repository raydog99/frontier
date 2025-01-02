open Torch

module Distributions : sig
  val normal : Tensor.t -> Tensor.t -> Tensor.t
end

module Payoffs : sig
  val forward_start_call : Tensor.t -> Tensor.t -> Tensor.t
end

module Wasserstein : sig
  val distance : int -> Tensor.t -> Tensor.t -> Tensor.t
  val adapted_distance : int -> Tensor.t -> Tensor.t -> Tensor.t
  val ball : Tensor.t -> Tensor.t -> (Tensor.t -> bool)
  val adapted_ball : Tensor.t -> Tensor.t -> (Tensor.t -> bool)
end

module Martingale : sig
  val constraint : Tensor.t -> Tensor.t -> bool
  val project : Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
end

module Marginal : sig
  val constraint : Tensor.t -> Tensor.t -> bool
  val project : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
end

module Hedging : sig
  val dynamic : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
  val static : (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t
  val semi_static : Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> Tensor.t
end

val print_tensor : Tensor.t -> unit
val run_experiment : [`BlackScholes | `Bachelier] -> Tensor.t -> Tensor.t -> unit
val compare_sensitivities : [`BlackScholes | `Bachelier] -> float list -> float list -> unit
val error_analysis : [`BlackScholes | `Bachelier] -> float -> float -> int -> unit