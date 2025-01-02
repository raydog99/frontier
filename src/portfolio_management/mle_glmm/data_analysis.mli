open Torch

val model_comparison : Glmm.t list -> (Glmm.t * float) list
(** [model_comparison models] compares models using AIC *)

val fixed_effects_summary : 
  Glmm.t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t
(** [fixed_effects_summary model] returns (estimates, SE, z-stats, p-values) *)

val residual_analysis : Glmm.t -> Algo.state -> 
  Tensor.t * Tensor.t * Tensor.t
(** [residual_analysis model state] returns (pearson, deviance, leverage) residuals *)

val influence_measures : Glmm.t -> Algo.state -> Tensor.t
(** [influence_measures model state] computes Cook's distances *)

val diagnostic_plots : Glmm.t -> Algo.state -> unit
(** [diagnostic_plots model state] generates diagnostic plots *)