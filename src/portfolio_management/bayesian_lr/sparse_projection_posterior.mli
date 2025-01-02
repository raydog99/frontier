open Torch

val fit : 
  Tensor.t -> Tensor.t -> Type.projection_config -> int -> 
  Type.posterior_sample array array * Type.convergence_diagnostics array
(** [fit x y config n_chains] fits the model with multiple chains *)

val fit_cv :
  Tensor.t -> Tensor.t -> Type.projection_config -> int ->
  Type.posterior_sample array array * Type.convergence_diagnostics array * Type.cv_result list
(** [fit_cv x y base_config k] fits model with cross-validation *)

val fit_adaptive :
  Tensor.t -> Tensor.t -> Type.projection_config -> Type.posterior_sample list
(** [fit_adaptive x y base_config] fits model with adaptive MCMC *)

val fit_parallel :
  Tensor.t -> Tensor.t -> Type.projection_config -> int ->
  Parallel.chain_state array * Type.convergence_diagnostics array
(** [fit_parallel x y config n_chains] fits model with parallel chains *)

val test_parameters :
  Type.posterior_sample list -> int list -> float list -> float -> Type.hypothesis_test list
(** [test_parameters samples parameters null_values alpha] performs hypothesis tests *)

val test_joint_hypothesis :
  Type.posterior_sample list -> int list -> float list -> float -> Type.joint_test
(** [test_joint_hypothesis samples indices null_values alpha] performs joint test *)

val compare_models :
  Type.posterior_sample list -> Tensor.t -> Tensor.t -> Type.model_comparison
(** [compare_models samples x y] computes model comparison metrics *)