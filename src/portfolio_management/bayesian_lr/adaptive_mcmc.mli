open Torch

val acceptance_prob :
  Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> 
  float -> Type.projection_config -> float
(** [acceptance_prob x y theta_new theta_old sigma config] computes acceptance probability *)

val adapt_step_size : Type.adaptive_config -> float -> unit
(** [adapt_step_size config acc_rate] adapts step size based on acceptance rate *)

val run : Tensor.t -> Tensor.t -> Type.adaptive_config -> Type.posterior_sample list
(** [run x y config] runs adaptive MCMC sampling *)