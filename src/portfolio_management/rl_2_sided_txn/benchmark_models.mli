open Torch

module MeanVariance : sig
  type t
  val create : int -> t
  val optimize : t -> Tensor.t -> Tensor.t -> float -> unit
  val update : t -> Tensor.t -> unit
  val calculate_portfolio_return : Tensor.t -> Tensor.t -> float
  val calculate_portfolio_risk : Tensor.t -> Tensor.t -> float
end

module MeanAbsoluteDeviation : sig
  type t
  val create : int -> t
  val optimize : t -> Tensor.t -> float -> unit
  val update : t -> Tensor.t -> unit
end

module ConditionalValueAtRisk : sig
  type t
  val create : int -> float -> t
  val optimize : t -> Tensor.t -> float -> unit
  val update : t -> Tensor.t -> unit
end
