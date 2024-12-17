open Types

type sample_function = {
  f: Tensor.t -> float * Tensor.t;
  is_clean: bool;
}

type optimization_params = {
  epsilon: float;       (** ε-contamination level *)
  sigma: float;        (** Covariance bound *)
  beta_bar: float option;  (** Smoothness parameter *)
  dimension: int;      (** Problem dimension *)
  diameter: float;     (** Domain diameter D *)
  tau: float;         (** Confidence parameter *)
}

type error_bounds = {
  corruption_error: float;    (** O(Dσ√ε) term *)
  statistical_error: float;   (** O(σD√(d/n)) term *)
}

type optimization_result = {
  solution: Tensor.t;
  value: float;
  bounds: error_bounds;
  iterations: int;
}

val project_onto_domain : Tensor.t -> float -> Tensor.t
val l2_distance : Tensor.t -> Tensor.t -> float
val compute_covariance : 
  Tensor.t list -> Tensor.t -> Types.optimization_params -> Tensor.t
  
val verify_covariance_bound : Tensor.t -> float -> bool
val estimate_gradient : 
  Tensor.t list -> Types.optimization_params -> Tensor.t
val generate : Types.optimization_params -> Tensor.t list
val optimize : 
  params:Types.optimization_params -> 
  functions:Types.sample_function list -> 
  Types.optimization_result