open Torch

module type SDE = sig
  val f : Tensor.t -> float -> Tensor.t
  val g : float -> float
  val sample : Tensor.t -> float -> Tensor.t -> Tensor.t
end

module VE_SDE : SDE
module VP_SDE : SDE