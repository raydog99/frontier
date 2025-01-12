open Torch

type model_params = {
  beta: tensor;              (** Regression coefficients *)
  gamma: float;             (** Location expansion parameter *)
  delta: float;             (** Scale expansion parameter *)
  prior_var: tensor;        (** Prior variance matrix *)
}

type latent_vars = {
  z: tensor;                (** Latent utilities *)
  omega: tensor;            (** PG variables *)
  z_tilde: tensor option;   (** Expanded utilities *)
}

type model_state = {
  params: model_params;
  latent: latent_vars;
}

type chain_stats = {
  acceptance_rates: float array;
  log_posterior: float array;
  convergence_r: float option;
}

type binomial_obs = {
  y: tensor;  (** Successes *)
  n: tensor;  (** Number of trials *)
}

type binomial_latent = {
  w: tensor;
  v: tensor;
  omega_w: tensor;
  omega_v: tensor;
  w_tilde: tensor option;
  v_tilde: tensor option;
}

type bin_state = {
  latent: binomial_latent;
  params: model_params;
}

type category_latent = {
  z_k: tensor;
  omega_k: tensor;
  z_tilde_k: tensor option;
}

type mnl_latent = {
  utilities: tensor;
  category_vars: category_latent array;
}

type mnl_params = {
  beta: tensor array;
  gamma: float array;
  delta: float array;
  prior_var: tensor;
}

type mnl_state = {
  latent: mnl_latent;
  params: mnl_params;
}

val stable_log1p : tensor -> float
(** Numerically stable implementation of log1p *)

val stable_sigmoid : tensor -> float
(** Numerically stable implementation of sigmoid function *)

val stable_rsqrt : tensor -> tensor
(** Numerically stable implementation of reciprocal square root *)

val stable_softmax : tensor -> tensor
(** Numerically stable implementation of softmax function *)

val safe_cholesky : tensor -> tensor
(** Cholesky decomposition with automatic jitter adjustment *)

val symmetrize : tensor -> tensor
(** Ensure matrix symmetry by averaging with transpose *)

val sample_pg : b:float -> c:float -> int list -> tensor
(** Sample from Pólya-Gamma distribution *)

val sample_conditional_omega : z:tensor -> tensor
(** Sample auxiliary omega variables conditional on z *)

val sample_truncated_utility : 
  mu:float -> 
  sigma:float -> 
  lower:float -> 
  upper:float -> 
  max_tries:int -> 
  float
(** Sample from truncated normal distribution *)

val sample_binary_utility :
  x_beta:tensor ->
  y:tensor ->
  n:int ->
  tensor
(** Sample binary latent utilities *)

val beta_function : float -> float -> float
(** Compute beta function *)

val log_density : float -> float -> float -> float
(** Compute log density of beta prime distribution *)

val sample_gli : nu:float -> loc:float -> scale:float -> int list -> tensor
(** Sample from generalized logistic type I distribution *)

val sample_glii : nu:float -> loc:float -> scale:float -> int list -> tensor
(** Sample from generalized logistic type II distribution *)

module JointPosterior : sig
  type conditional_params = {
    mean: tensor;
    precision: tensor;
    bounds: float * float;
  }

  val calc_joint_log_posterior : 
    x:tensor -> y:tensor -> state:model_state -> float
  (** Calculate log joint posterior density *)

  val get_conditional_beta :
    x:tensor ->
    z:tensor ->
    omega:tensor ->
    prior_var:tensor ->
    conditional_params
  (** Get conditional distribution for beta *)

  val get_conditional_gamma :
    z_tilde:tensor ->
    y:tensor ->
    conditional_params
  (** Get conditional distribution for gamma *)

  val get_conditional_delta :
    z:tensor ->
    omega:tensor ->
    conditional_params
  (** Get conditional distribution for delta *)
end

val expand_location :
  z:tensor -> gamma:float -> y:tensor -> tensor * tensor * tensor
(** Expand location parameter *)

val expand_scale :
  z:tensor -> delta:float -> omega:tensor -> tensor * tensor
(** Expand scale parameter *)

val expand_parameters :
  state:model_state -> y:tensor -> model_state
(** Combined parameter expansion *)

module MCMCKernel : sig
  type kernel_stats = {
    log_posterior: float;
    acceptance_rate: float;
    step_size: float;
  }

  val transition :
    state:model_state ->
    x:tensor ->
    y:tensor ->
    (model_state * kernel_stats, string) result
  (** Single MCMC transition step *)
end

val calc_offset :
  x:tensor ->
  beta:tensor array ->
  k:int ->
  n_categories:int ->
  tensor
(** Calculate multinomial offset term *)

val sample_utilities :
  x:tensor ->
  beta:tensor array ->
  y:tensor ->
  n_categories:int ->
  tensor
(** Sample multinomial utilities *)

val calc_utility_gaps :
  utilities:tensor ->
  n_categories:int ->
  tensor array
(** Calculate utility gaps between categories *)

val sample_dual_latents :
  x:tensor ->
  beta:tensor ->
  obs:binomial_obs ->
  tensor * tensor * tensor * tensor
(** Sample dual latent variables for binomial model *)

val expand_utilities :
  w:tensor ->
  v:tensor ->
  omega_w:tensor ->
  omega_v:tensor ->
  gamma:float ->
  delta:float ->
  obs:binomial_obs ->
  (tensor * tensor) * (tensor * tensor) * (tensor * tensor) * (float * float)
(** Expand utilities for binomial model *)

module MCMCEngine : sig
  val initialize : x:tensor -> y:tensor -> model_state
  (** Initialize MCMC sampler *)

  val calc_ess : float list -> float
  (** Calculate effective sample size *)

  val calc_rhat : chains:model_state list list -> float
  (** Calculate R-hat convergence diagnostic *)
end

module Diagnostics : sig
  type sampler_stats = {
    acceptance_rates: float array;
    effective_samples: float array;
    r_hat: float option;
    runtime: float;
  }

  val calc_acceptance_rates : model_state list -> float array
  (** Calculate MCMC acceptance rates *)
end

module UPGG : sig
  type model = 
    | Binary
    | Multinomial of int
    | Binomial

  type config = {
    n_warmup: int;           (** Number of warmup iterations *)
    n_iter: int;             (** Number of sampling iterations *)
    n_chains: int;           (** Number of parallel chains *)
    target_acceptance: float; (** Target acceptance rate *)
    rare_threshold: float;   (** Threshold for rare categories *)
  }

  val default_config : config
  (** Default MCMC configuration *)

  val sample :
    ?config:config ->
    x:tensor ->
    y:tensor ->
    model ->
    model_state list list * Diagnostics.sampler_stats
  (** Main sampling function *)
end