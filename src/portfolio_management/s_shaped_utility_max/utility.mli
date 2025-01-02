open Torch

module type Utility = sig
  val evaluate : Tensor.t -> Tensor.t
  val derivative : Tensor.t -> Tensor.t
end

module PowerUtility : sig
  val create : float -> (module Utility)
end

module LogUtility : sig
  val create : unit -> (module Utility)
end

module ExponentialUtility : sig
  val create : float -> (module Utility)
end

module SShapedUtility : sig
  val create : (module Utility) -> (module Utility) -> float -> (module Utility)
end

module ConcaveEnvelope : sig
  val create : (module Utility) -> (module Utility) -> float -> (module Utility)
end