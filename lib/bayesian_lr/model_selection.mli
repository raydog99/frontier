open Torch

val get_support : Tensor.t -> Tensor.t
(** [get_support theta] converts tensor to binary support vector *)

val compute_tpr : Tensor.t -> Tensor.t -> float
(** [compute_tpr true_support pred_support] computes True Positive Rate *)

val compute_fdp : Tensor.t -> Tensor.t -> float
(** [compute_fdp true_support pred_support] computes False Discovery Proportion *)

val compute_mcc : Tensor.t -> Tensor.t -> float
(** [compute_mcc true_support pred_support] computes Matthews Correlation Coefficient *)

val evaluate_selection : 
  Type.posterior_sample list -> Tensor.t -> Type.model_metrics
(** [evaluate_selection samples true_theta] computes all selection metrics *)

val get_top_variables : Type.posterior_sample list -> int -> Tensor.t
(** [get_top_variables samples k] returns indices of top k most frequently selected variables *)