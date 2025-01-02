open Torch

val gram_matrix : Tensor.t -> Tensor.t
(** Compute X'X *)

val cross_product : Tensor.t -> Tensor.t -> Tensor.t
(** Compute X'y *)

val solve : Tensor.t -> Tensor.t -> Tensor.t
(** Solve linear system using SVD *)