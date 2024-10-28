open Torch

val create_folds : 
  Tensor.t -> Tensor.t -> int -> Type.cv_fold array
(** [create_folds x y k] creates k-fold split *)

val grid_search :
  Tensor.t -> Tensor.t -> Type.projection_config -> int -> float list -> 
  Type.cv_result list
(** [grid_search x y config k lambda_grid] performs grid search for optimal lambda *)

val find_optimal_lambda :
  Tensor.t -> Tensor.t -> Type.projection_config -> int -> Type.cv_result list
(** [find_optimal_lambda x y config k] finds optimal lambda using cross-validation *)