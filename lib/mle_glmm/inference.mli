open Torch

val standard_errors : Glmm.t -> Algo.state -> Tensor.t
(** [standard_errors model state] computes standard errors of parameters *)

val confidence_intervals : 
  Glmm.t -> Algo.state -> float -> Tensor.t * Tensor.t
(** [confidence_intervals model state alpha] computes (lower, upper) CIs *)

val wald_test : 
  Glmm.t -> Algo.state -> Tensor.t -> Tensor.t * float
(** [wald_test model state contrast] computes test statistic and p-value *)

val information_criteria : 
  Glmm.t -> Algo.state -> float * float
(** [information_criteria model state] computes (AIC, BIC) *)

val likelihood_ratio_test :
  Glmm.t -> Glmm.t -> float * float
(** [likelihood_ratio_test model1 model2] computes LRT stat and p-value *)

val score_test :
  Glmm.t -> Algo.state -> Tensor.t -> float * float
(** [score_test model state contrast] computes score test stat and p-value *)