open Torch

type dimensions = {
  p_o : int;  (* Dimension of outside regressors *)
  p_c : int;  (* Dimension of common regressors *)
  q : int;    (* Dimension of common variables *)
}

type variable_type = 
  | Outside   (* X_o *)
  | Common    (* X_c *)
  | CommonVar (* W but not in X_c *)

type dataset = {
  features : Tensor.t;
  target : Tensor.t;
  common_vars : Tensor.t option;
}

type model_params = {
  lambda : float;
  include_common : bool;
}

val compute_residuals : Tensor.t -> Tensor.t -> Tensor.t
val partial_residuals : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
val mean : Tensor.t -> Tensor.t
val var : Tensor.t -> Tensor.t
val cov : Tensor.t -> Tensor.t -> Tensor.t
val quantile : Tensor.t -> float -> float
val compute_eta_d : dims:dimensions -> Tensor.t -> Tensor.t -> Tensor.t
val compute_support_function : dims:dimensions -> Tensor.t -> Tensor.t -> Tensor.t -> float
val compute_sharp_bounds : dataset -> float
val compute_outer_bound : dataset -> (Tensor.t -> Tensor.t) -> float
val decompose_variables : Tensor.t -> Tensor.t -> Tensor.t -> 
  (Tensor.t * Tensor.t) * (Tensor.t * Tensor.t)
val conditional_quantile_function : Tensor.t -> Tensor.t -> float -> Tensor.t
val compute_sharp_bounds : dataset -> float
val compute_influence_functions : dataset -> Tensor.t * Tensor.t * Tensor.t * Tensor.t
val compute_asymptotic_variance : dataset -> float -> float
val compute_clt_distribution : dataset -> float -> float * float
val confidence_interval : ?alpha:float -> dataset -> float -> float * float