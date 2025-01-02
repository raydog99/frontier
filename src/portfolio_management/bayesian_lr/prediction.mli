open Torch

val predict_samples : 
  Tensor.t -> Type.posterior_sample list -> float -> Type.prediction
(** [predict_samples x_new samples alpha] makes predictions with uncertainty *)

val compute_metrics : 
  Tensor.t -> Tensor.t -> float * float * float
(** [compute_metrics y_true y_pred] computes prediction metrics (MSE, MAE, R²) *)