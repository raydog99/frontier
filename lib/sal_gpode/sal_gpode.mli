open Torch

type gp_params = {
  lengthscales: Tensor.t;
  signal_variance: float;
  noise_variance: float;
}

type sparse_gp = {
  inducing_inputs: Tensor.t;
  inducing_outputs: Tensor.t;
  variational_mean: Tensor.t;
  variational_cov: Tensor.t;
  params: gp_params;
}

type gpode_model = {
  gp: sparse_gp;
  integration_times: Tensor.t;
  dt: float;
}

val rbf_kernel : params:gp_params -> x1:Tensor.t -> x2:Tensor.t -> Tensor.t
val kernel_matrix : params:gp_params -> x1:Tensor.t -> x2:Tensor.t -> Tensor.t

val nystrom_approximation : 
  kernel:(Tensor.t -> Tensor.t -> Tensor.t) ->
  inducing_points:Tensor.t ->
  x:Tensor.t ->
  Tensor.t

val random_feature_approximation : 
    num_features:int ->
    input_dim:int ->
    params:gp_params ->
    (Tensor.t -> Tensor.t)
    
val sample_function : 
    phi:(Tensor.t -> Tensor.t) ->
    weights:Tensor.t ->
    Tensor.t ->
    Tensor.t

module Integration : sig
  type integrator = [ `RK4 | `Adaptive | `DormandPrince ]
  
  val integrate_trajectory :
    model:gpode_model ->
    x0:Tensor.t ->
    t_span:float ->
    integrator:integrator ->
    tol:float ->
    Tensor.t list
end

module Information : sig
  type t = [ `MutualInfo | `Entropy | `Variance | `ELBO ]
  
  val calculate_entropy : trajectories:Tensor.t -> float
  val calculate_mutual_info : model:gpode_model -> x0:Tensor.t -> num_samples:int -> float
  val calculate_variance : model:gpode_model -> x0:Tensor.t -> float
  val calculate_elbo : model:gpode_model -> x0:Tensor.t -> trajectories:Tensor.t -> float
end

module Safety : sig
  type constraint_type = [ `Hard | `Soft of float ]
  
  type config = {
    x_min: Tensor.t;
    x_max: Tensor.t;
    constraint_type: constraint_type;
    prob_threshold: float;
  }
  
  val evaluate_safety :
    model:gpode_model ->
    x0:Tensor.t ->
    config:config ->
    num_samples:int ->
    float
    
  val compute_gradient :
    model:gpode_model ->
    x:Tensor.t ->
    config:config ->
    Tensor.t
end

module TimeVaryingSafety : sig
  type constraint_fn = float -> Tensor.t * Tensor.t
  
  val evaluate_time_varying_safety :
    model:gpode_model ->
    x0:Tensor.t ->
    constraint_fn:constraint_fn ->
    num_samples:int ->
    float
    
  val compute_time_varying_gradient :
    model:gpode_model ->
    x:Tensor.t ->
    t:float ->
    constraint_fn:constraint_fn ->
    Tensor.t
end

module ModelUpdate : sig
  val update_variational_distribution :
    model:gpode_model ->
    new_data:(Tensor.t * Tensor.t) ->
    num_iters:int ->
    gpode_model
    
  val optimize_hyperparams :
    model:gpode_model ->
    learning_rate:float ->
    num_iters:int ->
    gpode_model
end

module SafeOptimization : sig
  type config = {
    num_samples: int;
    max_iter: int;
    learning_rate: float;
    safety_config: Safety.config;
  }
  
  val optimize :
    model:gpode_model ->
    x_init:Tensor.t ->
    config:config ->
    Tensor.t option
    
  val optimize_with_time_varying_constraints :
    model:gpode_model ->
    x_init:Tensor.t ->
    constraint_fn:TimeVaryingSafety.constraint_fn ->
    config:config ->
    Tensor.t
end

val create_model :
  inducing_points:Tensor.t ->
  params:gp_params ->
  dt:float ->
  t_span:float ->
  gpode_model

val safe_active_learning :
  model:gpode_model ->
  config:SafeOptimization.config ->
  (gpode_model * float list, string) result