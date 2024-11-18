open Torch

module Types : sig
  type dimension = {
    input_dim: int;
    batch_size: int;
  }

  type diffusion_params = {
    dt: float;
    total_time: float;
    epsilon: float;
  }

  type score_config = {
    n_samples: int;
    step_size: float;
    n_steps: int;
    threshold: float;
    switch_time: float;
    l2_reg: float;
    grad_clip: float option;
    temperature: float;
  }

  type sample_stats = {
    acceptance_rate: float;
    ess: float;
    grad_norm: float;
    kl_estimate: float option;
  }

  type chain_state = {
    position: Tensor.t;
    log_prob: Tensor.t;
    score: Tensor.t;
    stats: sample_stats;
  }

  type monitor_config = {
    check_interval: int;
    target_r_hat: float;
    min_n_eff: float;
  }
end

module Target : sig
  module Make : functor (P : sig
    val mean : float array
    val std : float
  end) -> sig
    val log_density : Tensor.t -> Tensor.t
    val grad_log_density : Tensor.t -> Tensor.t
  end

  val make_mixture : means:float array -> std:float -> 
    (module sig
      val log_density : Tensor.t -> Tensor.t
      val grad_log_density : Tensor.t -> Tensor.t
    end)
end

module OU_process : sig
  val transition_kernel : t:float -> x:Tensor.t -> x0:Tensor.t -> Tensor.t
  val forward_sde : t:float -> x:Tensor.t -> Tensor.t * Tensor.t
  val reverse_sde : t:float -> x:Tensor.t -> score:Tensor.t -> Tensor.t * Tensor.t
  val scaled_reverse_sde : 
    t:float -> x:Tensor.t -> score:Tensor.t -> temperature:float -> Tensor.t * Tensor.t
end

module Score_estimator : sig
  val estimate_score :
    < log_density : Tensor.t -> Tensor.t;
      grad_log_density : Tensor.t -> Tensor.t; .. > ->
    t:float -> 
    x:Tensor.t -> 
    config:Types.score_config -> 
    Tensor.t

  val get_acceptance_rate : unit -> float
end

module RDMC : sig
  val sample :
    < log_density : Tensor.t -> Tensor.t;
      grad_log_density : Tensor.t -> Tensor.t; .. > ->
    config:(Types.diffusion_params * Types.score_config) ->
    init_sample:Tensor.t ->
    Tensor.t

  val sample_gaussian_mixture :
    means:float array ->
    std:float ->
    config:(Types.diffusion_params * Types.score_config) ->
    init_sample:Tensor.t ->
    Tensor.t
end