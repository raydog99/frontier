open Torch

val predict_random_effects : 
  Glmm.t -> Algo.state -> Tensor.t
(** [predict_random_effects model state] predicts random effects *)

val prediction_variance : 
  Glmm.t -> Tensor.t -> Tensor.t
(** [prediction_variance model gamma] computes prediction variance *)

val predict_response : 
  Glmm.t -> Algo.state -> Tensor.t -> Tensor.t -> Tensor.t
(** [predict_response model state new_x new_z] predicts responses for new data *)

val prediction_intervals :
  Glmm.t -> Algo.state -> Tensor.t -> Tensor.t -> float ->
  Tensor.t * Tensor.t
(** [prediction_intervals model state new_x new_z alpha] computes prediction intervals *)

val cross_validate :
  Glmm.t -> int -> float * float
(** [cross_validate model k] performs k-fold cross-validation returning (mean_error, std_error) *)