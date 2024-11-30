open Torch

type log_likelihood = Tensor.t -> Tensor.t
type log_prior = Tensor.t -> Tensor.t
type markov_kernel = Tensor.t -> Tensor.t -> Tensor.t

val uniform_subsample : size:int -> n:int -> int array
val linear_detrend : float array -> float * float
val estimate : 
  coreset:Coreset.t -> 
  states:Tensor.t array -> 
  subsample_indices:int array -> 
  Tensor.t

module Coreset : sig
  type t = private {
    weights: Tensor.t;
    log_likelihoods: log_likelihood array;
    log_prior: log_prior;
  }

  val create : 
    weights:Tensor.t -> 
    log_likelihoods:log_likelihood array -> 
    log_prior:log_prior -> 
    t

  val log_posterior : t -> Tensor.t -> Tensor.t
end

module HotDog : sig
  type state = private {
    v: Tensor.t;           (* Second moment estimate *)
    m: Tensor.t;           (* First moment estimate *)
    d: Tensor.t;           (* Maximum distance *)
    c: int;                      (* Iteration counter *)
    h: bool;                     (* Hot-start flag *)
    log_potentials: float array; (* For hot-start test *)
  }

  val create_state : Tensor.t -> state

  val update : 
    beta1:float ->
    beta2:float ->
    epsilon:float ->
    r:float ->
    state:state ->
    weights:Tensor.t ->
    grad:Tensor.t ->
    state * Tensor.t

  val hot_start_test : float array -> bool
end

module CoresetMCMC : sig
  type config = {
    n_chains: int;              (* Number of MCMC chains *)
    subsample_size: int;        (* Size of data subsample *)
    max_iters: int;            (* Maximum iterations *)
    beta1: float;              (* First moment decay *)
    beta2: float;              (* Second moment decay *)
    epsilon: float;            (* Numerical stability *)
    r: float;                  (* Initial learning rate *)
  }

  val default_config : config

  val run : 
    config ->
    Coreset.t ->
    Tensor.t array ->  (* Initial states *)
    Tensor.t * Tensor.t array list  (* Final weights and samples *)
end