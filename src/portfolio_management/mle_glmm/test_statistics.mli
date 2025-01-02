open Torch

val fisher_information : Glmm.t -> Tensor.t
(** [fisher_information model] computes Fisher Information matrix *)

val compute_p_value : Tensor.t -> int -> float
(** [compute_p_value stat df] computes p-value from chi-square distribution *)

val likelihood_ratio_stat : Glmm.t -> Glmm.t -> Tensor.t
(** [likelihood_ratio_stat model1 model2] computes LR statistic *)

val score_stat : Glmm.t -> Glmm.t -> Tensor.t -> Tensor.t
(** [score_stat model1 model2 b] computes score statistic *)

val wald_stat : Glmm.t -> Glmm.t -> Tensor.t -> Tensor.t
(** [wald_stat model1 model2 b] computes generalized Wald statistic *)