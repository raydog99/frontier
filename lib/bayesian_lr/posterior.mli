open Torch

val sample_posterior : 
  Tensor.t -> Tensor.t -> Type.projection_config -> float -> Tensor.t
(** [sample_posterior x y config sigma] samples from multivariate normal posterior *)

val sample_posterior_distributed : 
  Type.distributed_data -> Type.projection_config -> float -> Tensor.t
(** [sample_posterior_distributed data config sigma] samples using distributed 
    sufficient statistics *)