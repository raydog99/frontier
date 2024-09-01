open Torch

module GBM : sig
  type tree
  type model

  val predict : model -> Tensor.t -> float
  val train : Tensor.t -> Tensor.t -> int -> int -> float -> model
end

module NN : sig
  type model

  val forward : model -> Tensor.t -> Tensor.t
  val train : Tensor.t -> Tensor.t -> int list -> float -> int -> model
  val predict : model -> Tensor.t -> Tensor.t
end

module GPR : sig
  type model

  val rbf_kernel : float -> (Tensor.t -> Tensor.t -> Tensor.t)
  val predict : model -> Tensor.t -> Tensor.t * Tensor.t
  val train : Tensor.t -> Tensor.t -> float -> float -> model
end