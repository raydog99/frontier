open Torch

val log_likelihood : 
  Tensor.t -> Tensor.t -> float -> float
(** [log_likelihood y pred sigma] computes log likelihood *)

val compute_dic : 
  Type.posterior_sample list -> Tensor.t -> Tensor.t -> Type.model_comparison
(** [compute_dic samples x y] computes DIC *)

val compute_waic : 
  Type.posterior_sample list -> Tensor.t -> Tensor.t -> Type.model_comparison
(** [compute_waic samples x y] computes WAIC *)