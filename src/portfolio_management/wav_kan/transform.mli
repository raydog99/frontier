open Torch

module CWT : sig
  type transform_result = {
    coefficients: Tensor.t;
    scales: Tensor.t;
    frequencies: Tensor.t;
    energy: Tensor.t;
  }

  val transform : Tensor.t -> (float -> float) -> transform_result
  val reconstruct : transform_result -> (float -> float) -> float -> Tensor.t
end

module DWT : sig
  type coefficients = {
    approximation: Tensor.t;
    details: Tensor.t list;
    level: int;
  }

  val decompose : Tensor.t -> int -> coefficients
  val reconstruct : coefficients -> Tensor.t
end