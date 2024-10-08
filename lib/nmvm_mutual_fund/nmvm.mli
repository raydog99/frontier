open Torch

type t

type distribution = 
  | Normal of float * float
  | LogNormal of float * float
  | Gamma of float * float
  | InverseGaussian of float * float

val create : Tensor.t -> Tensor.t -> Tensor.t -> distribution -> t
val sample : t -> Tensor.t
val expected_return : t -> Tensor.t
val covariance : t -> Tensor.t
val infinity_number : t -> float
val generate_samples : t -> int -> Tensor.t list
val estimate_parameters : Tensor.t list -> t