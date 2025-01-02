open Torch

module DiffusionSpace : sig
  type t = {
    heat_kernel: HeatKernel.t;
    time: float;
    manifold: Manifold.t;
  }

  val create : Manifold.t -> HeatKernel.t -> float -> t
  val inner_product : t -> Tensor.t -> Tensor.t -> float
  val embed : t -> Tensor.t -> Tensor.t
  val embed_adjoint : t -> Tensor.t -> Tensor.t
end

module IntegralOperator : sig
  type t = {
    kernel: HeatKernel.t;
    points: Tensor.t;
  }

  val create : HeatKernel.t -> Tensor.t -> t
  val apply : t -> Tensor.t -> Tensor.t
  val operator_norm : t -> float
  val power_iterate : t -> int -> float -> float
end

module PowerSpace : sig
  type t = {
    alpha: float;
    eigenvalues: Tensor.t;
    eigenvectors: Tensor.t;
    time: float;
  }

  val create : float -> Tensor.t -> Tensor.t -> float -> t
  val norm : t -> Tensor.t -> float
  val project : t -> Tensor.t -> Tensor.t
end