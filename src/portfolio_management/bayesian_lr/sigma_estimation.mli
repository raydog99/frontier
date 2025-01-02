open Torch

val estimate_sigma : 
  Tensor.t -> Tensor.t -> Tensor.t -> float
(** [estimate_sigma x y theta_star] computes consistent sigma estimate *)

val estimate_sigma_distributed : 
  Type.distributed_data -> Tensor.t -> float
(** [estimate_sigma_distributed data theta_star] computes sigma estimate with distributed data *)

val sample_sigma : 
  Tensor.t -> Tensor.t -> Tensor.t -> int -> float
(** [sample_sigma x y theta_star n] samples from immersion posterior for sigma *)

val sample_sigma_distributed : 
  Type.distributed_data -> Tensor.t -> float
(** [sample_sigma_distributed data theta_star] samples sigma with distributed data *)