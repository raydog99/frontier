open Torch

val matern_covariance : Types.matern_params -> Tensor.t -> Tensor.t
(** [matern_covariance params dist_mat] computes Matérn covariance matrix *)

val compute_covariance : Types.model_spec -> Tensor.t -> Tensor.t
(** [compute_covariance spec dist_mat] computes covariance matrix based on model specification *)

val derivative_matern : Types.matern_params -> Tensor.t -> Tensor.t
(** [derivative_matern params dist_mat] computes derivative of Matérn covariance *)

val derivative_covariance : Types.model_spec -> Tensor.t -> Tensor.t
(** [derivative_covariance spec dist_mat] computes derivative of covariance matrix *)