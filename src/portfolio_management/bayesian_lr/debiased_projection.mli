open Torch

val compute_residuals : 
  Tensor.t -> int -> float -> Tensor.t
(** [compute_residuals x j lambda_x] computes LASSO residuals R_j *)

val debiased_projection : 
  Tensor.t -> Tensor.t -> Tensor.t -> Type.projection_config -> Tensor.t
(** [debiased_projection x theta theta_star config] computes debiased projection map *)

val debiased_projection_distributed : 
  Type.distributed_data -> Tensor.t -> Tensor.t -> Type.projection_config -> Tensor.t
(** [debiased_projection_distributed data theta theta_star config] computes debiased 
    projection using distributed sufficient statistics *)