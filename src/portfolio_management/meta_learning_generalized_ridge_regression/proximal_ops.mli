open Torch

val soft_threshold :
  x:Tensor.t ->
  lambda:float ->
  Tensor.t
(** Soft thresholding operator *)

val matrix_l1_prox :
  matrix:Tensor.t ->
  lambda:float ->
  Tensor.t
(** L1 proximal operator for matrices *)

val nuclear_prox :
  matrix:Tensor.t ->
  lambda:float ->
  Tensor.t
(** Nuclear norm proximal operator *)

val project_pd_cone :
  matrix:Tensor.t ->
  epsilon:float ->
  Tensor.t
(** Project onto positive definite cone *)