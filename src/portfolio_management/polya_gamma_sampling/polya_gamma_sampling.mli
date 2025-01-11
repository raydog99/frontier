open Torch

type model_params = {
  beta: tensor;
  gamma: float;
  delta: float;
  prior_var: tensor;
}

type latent_vars = {
  z: tensor;
  omega: tensor;
  z_tilde: tensor option;
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

(* Numerical Utils*)
val stable_log1p : float -> float
val stable_sigmoid : float -> float
val safe_cholesky : tensor -> tensor
val stable_rsqrt : tensor -> tensor
val symmetrize : tensor -> tensor
val stable_softmax : tensor -> tensor

(* PG Distribution*)
val sample_pg1_0 : unit -> float
val sample_pg : b:float -> c:float -> int array -> tensor
val sample_conditional_omega : z:tensor -> tensor

(* Utility Sampling *)
val sample_truncated_utility : mu:float -> sigma:float -> lower:float -> 
                              upper:float -> max_tries:int -> float
val sample_binary_utility : x_beta:tensor -> y:tensor -> n:int -> 
                           tensor

module JointPosterior : sig
  type conditional_params = {
    mean: tensor;
    precision: tensor;
    bounds: float * float;
  }

  val calc_joint_log_posterior : x:tensor -> y:tensor -> 
                                model_state -> float
  val get_conditional_beta : x:tensor -> z:tensor -> 
                           omega:tensor -> prior_var:tensor -> 
                           conditional_params
  val get_conditional_gamma : z_tilde:tensor -> y:tensor -> 
                            conditional_params
  val get_conditional_delta : z:tensor -> omega:tensor -> 
                            conditional_params
end

(* Parameter Expansion *)
val expand_location : z:tensor -> gamma:float -> y:tensor -> 
                     tensor * tensor * tensor
val expand_scale : z:tensor -> delta:float -> omega:tensor -> 
                  tensor * tensor
val expand_parameters : state:model_state -> y:tensor -> 
                       model_state

module MCMCKernel : sig
  type kernel_stats = {
    log_posterior: float;
    acceptance_rate: float;
    step_size: float;
  }

  val transition : state:model_state -> x:tensor -> y:tensor -> 
                  (model_state * kernel_stats, string) result
end

module MultinomialTypes : sig
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
end

val calc_offset : x:tensor -> beta:tensor array -> k:int -> 
                 n_categories:int -> tensor
val sample_utilities : x:tensor -> beta:tensor array -> 
                      y:tensor -> n_categories:int -> tensor
val calc_utility_gaps : utilities:tensor -> n_categories:int -> 
                       tensor array

module BinomialTypes : sig
  type binomial_obs = {
    y: tensor;
    n: tensor;
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
end

val sample_gli : nu:float -> loc:float -> scale:float -> int array -> 
                 tensor
val sample_glii : nu:float -> loc:float -> scale:float -> int array -> 
                  tensor
val sample_dual_latents : x:tensor -> beta:tensor -> 
                         obs:Binomialbinomial_obs -> 
                         tensor * tensor * tensor * tensor

val expand_utilities : w:tensor -> v:tensor -> 
                      omega_w:tensor -> omega_v:tensor -> 
                      gamma:float -> delta:float -> 
                      obs:Binomialbinomial_obs ->
                      (tensor * tensor) *
                      (tensor * tensor) *
                      (tensor * tensor) *
                      (float * float)

module MCMCEngine : sig
  val initialize : x:tensor -> y:tensor -> model_state
  val calc_ess : float list -> float
  val calc_rhat : chains:model_state list list -> float
end

module Diagnostics : sig
  type sampler_stats = {
    acceptance_rates: float array;
    effective_samples: float array;
    r_hat: float option;
    runtime: float;
  }

  val calc_acceptance_rates : model_state list -> float array
end

module UPGG : sig
  type model = 
    | Binary 
    | Multinomial of int
    | Binomial

  type config = {
    n_warmup: int;
    n_iter: int;
    n_chains: int;
    target_acceptance: float;
    rare_threshold: float;
  }

  val default_config : config

  val sample_model : ?config:config -> x:tensor -> y:tensor -> 
                    model -> model_state list list * Diagnostics.sampler_stats

  val sample : ?config:config -> x:tensor -> y:tensor -> model -> 
               model_state list list * Diagnostics.sampler_stats
end