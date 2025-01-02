open Torch

type weight_vector = Tensor.t

type model_params = {
  n : int;  (** Sample size *)
  m : int;  (** Resample size *)
  monte_carlo_b : int;  (** Monte Carlo size B *)
}

type nested_model = {
  x : Tensor.t;  (** Design matrix *)
  k : int;  (** Number of regressors *)
}

type optimization_result = {
  weights : Tensor.t;
  converged : bool;
  iterations : int;
  final_error : float;
}

type btma_result = {
  weights : Tensor.t;
  coefficients : Tensor.t;
  optimization_info : optimization_result;
  replications : int;
  valid_replications : int;
}

val validate : model_params -> bool
val default_params : int -> model_params

val safe_inverse : Tensor.t -> Tensor.t option
val pinverse : Tensor.t -> Tensor.t

val generate_sequence : x:Tensor.t -> nested_model list
val get_model : nested_model list -> int -> nested_model
val count : nested_model list -> int

val generate_resampling_matrix : n:int -> m:int -> Tensor.t
val calculate_ls_estimator : x_star:Tensor.t -> y_star:Tensor.t -> Tensor.t
val generate_valid_sample : x:Tensor.t -> y:Tensor.t -> m:int -> Tensor.t * Tensor.t

val run_replications : 
  x:Tensor.t -> 
  y:Tensor.t -> 
  params:model_params -> 
  models:nested_model list -> 
  Tensor.t

val optimize : 
  residual_matrix:Tensor.t -> 
  params:model_params -> 
  optimization_result

val project_onto_simplex : Tensor.t -> Tensor.t

val calculate_risk_decomposition : 
  x:Tensor.t -> 
  weights:Tensor.t -> 
  sigma_sq:float -> 
  Tensor.t * Tensor.t

val calculate_xi_n : 
  x:Tensor.t -> 
  y:Tensor.t -> 
  Tensor.t

val check_conditions : 
  x:Tensor.t -> 
  y:Tensor.t -> 
  params:model_params -> 
  bool * bool * bool * bool

val calculate_divergence : mu_hat:Tensor.t -> mu:Tensor.t -> Tensor.t

val asymptotic_optimality_condition : 
  x:Tensor.t -> 
  y:Tensor.t -> 
  weights:Tensor.t -> 
  sigma_sq:float -> 
  params:model_params -> 
  bool

val estimate : 
  x:Tensor.t -> 
  y:Tensor.t -> 
  params:model_params -> 
  btma_result

val calculate_h_omega : x:Tensor.t -> weights:Tensor.t -> Tensor.t

val multinomial_properties_condition : 
  pi_matrix:Tensor.t -> 
  n:int -> 
  m:int -> 
  bool * bool