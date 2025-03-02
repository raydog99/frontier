open Torch

(* Sample from a normal distribution *)
val normal_sample : mean:Tensor.t -> std:Tensor.t -> shape:int list -> Tensor.t

(* Log probability density of normal distribution *)
val normal_logpdf : Tensor.t -> mean:Tensor.t -> std:Tensor.t -> Tensor.t

(* MCMC *)
module MCMC : sig
  (* Metropolis-Hastings algorithm step *)
  val metropolis_step : 
    Tensor.t -> 
    proposal_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    log_prob_fn:(Tensor.t -> Tensor.t) -> 
    step_size:Tensor.t -> 
    Tensor.t * bool
  
  (* Random walk proposal *)
  val random_walk_proposal : Tensor.t -> Tensor.t -> Tensor.t
end

(* Replica Exchange Monte Carlo *)
module REMC : sig
  (* Parameters for REMC *)
  type params = {
    n_replicas: int;
    inverse_temperatures: Tensor.t;
    step_sizes: Tensor.t;
    burn_in: int;
    n_samples: int;
  }
  
  (* Create a set of inverse temperatures *)
  val create_temperatures : 
    min_temp:float -> 
    max_temp:float -> 
    n_replicas:int -> 
    Tensor.t
  
  (* Perform replica exchange between two replicas *)
  val exchange_replicas : 
    replica_i:Tensor.t -> 
    replica_j:Tensor.t -> 
    beta_i:Tensor.t -> 
    beta_j:Tensor.t -> 
    log_prob_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    Tensor.t * Tensor.t * bool
  
  (* Calculate Bayesian free energy difference *)
  val calculate_free_energy_diff : 
    samples:Tensor.t list -> 
    beta_i:Tensor.t -> 
    beta_j:Tensor.t -> 
    log_prob_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    Tensor.t
  
  (* REMC algorithm *)
  val run : 
    initial_samples:Tensor.t array -> 
    log_prob_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    proposal_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    params -> 
    Tensor.t list array
end

(* Sequential Monte Carlo Samplers *)
module SMCS : sig
  (* Parameters *)
  type params = {
    n_samples: int;
    n_mcmc_steps: int;
    step_sizes: Tensor.t;
  }
  
  (* Resample particles according to weights *)
  val resample : Tensor.t array -> float array -> Tensor.t array
  
  (* Calculate ESS (Effective Sample Size) *)
  val calculate_ess : float array -> float
  
  (* SMCS algorithm *)
  val run : 
    initial_samples:Tensor.t array -> 
    log_prior_fn:(Tensor.t -> Tensor.t) -> 
    log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
    proposal_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    params -> 
    Tensor.t list
end

(* Waste-free Sequential Monte Carlo *)
module WasteFree_SMC : sig
  (* Parameters *)
  type params = {
    n_samples: int;
    n_chains: int;
    n_mcmc_steps: int;
    step_sizes: Tensor.t;
  }
  
  (* Waste-free SMC algorithm *)
  val run : 
    initial_samples:Tensor.t array -> 
    log_prior_fn:(Tensor.t -> Tensor.t) -> 
    log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
    proposal_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    params -> 
    Tensor.t list
end

(* Robbins-Monro algorithm for step size adaptation *)
module RobbinsMonro : sig
  (* Parameters for Robbins-Monro *)
  type params = {
    c: float;
    n0: int;
    target_acceptance: float;
  }
  
  (* Update step size based on acceptance rate *)
  val update_step_size : 
    current_step_size:float -> 
    acceptance_rate:float -> 
    params:params -> 
    iteration:int -> 
    float
end

(* Free energy calculation utilities *)
val calculate_entropy : 
  samples:Tensor.t list -> 
  log_prob_fn:(Tensor.t -> Tensor.t) -> 
  float

val calculate_free_energy_diff : 
  samples:Tensor.t list -> 
  beta_i:float -> 
  beta_j:float -> 
  log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
  float

val thermodynamic_integration : 
  samples_at_betas:Tensor.t list array -> 
  betas:float array -> 
  log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
  float

(* Sequential Exchange Monte Carlo *)
module SEMC : sig
  (* Parameters *)
  type params = {
    n_samples: int;                (* Number of samples *)
    n_parallel: int;               (* Number of parallel chains *)
    target_exchange_rate: float;   (* Target exchange rate between temperatures *)
    step_size_params: {
      initial: float;              (* Initial step size *)
      adaptation_constant: float;  (* Robbins-Monro adaptation constant *)
      adaptation_offset: int;      (* Robbins-Monro adaptation offset *)
      target_acceptance: float;    (* Target acceptance rate for Metropolis steps *)
    };
  }
  
  (* Default parameters *)
  val default_params : params
  
  (* Determine next inverse temperature to maintain constant exchange rate *)
  val determine_next_beta : 
    current_beta:float -> 
    samples:Tensor.t array -> 
    log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
    target_exchange_rate:float -> 
    float
  
  (* Update step size using Robbins-Monro algorithm *)
  val update_step_size : 
    current_step_size:float -> 
    acceptance_rate:float -> 
    params:params -> 
    iteration:int -> 
    float
  
  (* Calculate weights for resampling *)
  val calculate_weights : 
    samples:Tensor.t array -> 
    log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
    beta_prev:float -> 
    beta_next:float -> 
    float array
  
  (* SEMC algorithm *)
  val run : 
    initial_samples:Tensor.t array -> 
    log_prior_fn:(Tensor.t -> Tensor.t) -> 
    log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
    proposal_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    ?params:params -> 
    ?verbose:bool -> 
    unit -> 
    Tensor.t list * float list
  
  (* Calculate Bayesian free energy *)
  val calculate_free_energy : 
    samples:Tensor.t list -> 
    betas:float list -> 
    log_prior_fn:(Tensor.t -> Tensor.t) -> 
    log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
    float
  
  (* multiple chains and check convergence *)
  val run_multiple_chains : 
    log_prior_fn:(Tensor.t -> Tensor.t) -> 
    log_likelihood_fn:(Tensor.t -> Tensor.t) -> 
    proposal_fn:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    initial_samples:Tensor.t array -> 
    n_chains:int -> 
    ?params:params -> 
    ?r_hat_threshold:float -> 
    ?verbose:bool -> 
    unit -> 
    Tensor.t list * Tensor.t list array * float list array * float array
end