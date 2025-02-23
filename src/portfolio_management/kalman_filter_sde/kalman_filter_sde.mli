open Torch

type state = Tensor.t
type observation = Tensor.t
type time = float

type parameters = {
  mu: float;      (* Drift/mean parameter *)
  theta: float;   (* Mean reversion level *)
  sigma: float;   (* Volatility parameter *)
  rho: float;     (* Correlation parameter *)
  kappa: float;   (* Mean reversion speed *)
  xi: float;      (* Volatility of volatility *)
}

(* Kalman Filter *)
module KalmanFilter : sig
  type t

  val create : 
    init_state:state ->
    init_cov:Tensor.t ->
    trans_mat:Tensor.t ->
    obs_mat:Tensor.t ->
    proc_noise:Tensor.t ->
    obs_noise:Tensor.t -> t

  val predict : t -> t
  val update : t -> observation -> t
end

(* Extended Kalman Filter *)
module ExtendedKalmanFilter : sig
  type t

  val create :
    init_state:state ->
    init_cov:Tensor.t ->
    proc_noise:Tensor.t ->
    obs_noise:Tensor.t ->
    state_trans:(state -> state) ->
    obs_func:(state -> observation) ->
    state_jac:(state -> Tensor.t) ->
    obs_jac:(state -> Tensor.t) -> t

  val predict : t -> t
  val update : t -> observation -> t
  val update_second_order : t -> observation -> t
end

(* Square Root Kalman Filter *)
module SquareRootKalmanFilter : sig
  type t

  val create :
    init_state:state ->
    init_cov:Tensor.t ->
    trans_mat:Tensor.t ->
    obs_mat:Tensor.t ->
    proc_noise:Tensor.t ->
    obs_noise:Tensor.t -> t

  val predict : t -> t
  val update : t -> observation -> t
  val update_invariant : t -> observation -> t
end

(* Particle Filter *)
module ParticleFilter : sig
  type t
  type particle = {
    state: state;
    weight: float;
  }

  type resampling_scheme =
    | Multinomial
    | Systematic
    | Stratified
    | Residual

  val create :
    n_particles:int ->
    init_state:state ->
    state_trans:(state -> state) ->
    obs_likelihood:(observation -> state -> float) ->
    resample_thresh:float ->
    resample_method:resampling_scheme -> t

  val predict_and_update : t -> observation -> t
  val estimate : t -> state
  val effective_sample_size : t -> float
end

(* Black-Karasinski Model *)
module BlackKarasinskiModel : sig
  type term_structure = {
    times: float array;
    rates: float array;
    volatilities: float array;
  }

  type t

  val create :
    mu:float ->
    theta:float ->
    sigma:float ->
    dt:float ->
    term_struct:term_structure -> t

  val step_short_rate : t -> float -> float
  val forward_rate : t -> float -> float -> float
  val simulate : t -> float -> int -> float list
  val create_ekf : t -> state -> ExtendedKalmanFilter.t
  val mle_estimate : float array -> float -> parameters
end

(* Jump Process *)
module JumpProcess : sig
  type jump_type =
    | PoissonJump of float * float
    | CompoundPoisson of float * (unit -> float)
    | VarianceGamma of float * float * float
    | NIG of float * float * float

  type levy_measure = {
    small_jumps: float -> float;
    large_jumps: float -> float;
    truncation: float;
  }

  type t

  val simulate_jump : t -> float
  val approximate_small_jumps : t -> float
  val simulate : t -> state -> int -> state list
end

(* Multivariate Jump Process *)
module MultivariateJumpProcess : sig
  type jump_correlation = {
    matrix: float array array;
    cholesky: float array array option;
  }

  type t

  val create :
    JumpProcess.t array ->
    float array array ->
    float -> t

  val simulate : t -> state array -> int -> float array array
end

(* Parameter Estimation *)
module ParameterEstimation : sig
  type estimation_method =
    | MaximumLikelihood
    | MethodOfMoments
    | GMM
    | MCMC of int

  type estimation_config = {
    method_type: estimation_method;
    learning_rate: float;
    tolerance: float;
    regularization: float;
    batch_size: int option;
  }

  module MomentEstimation : sig
    type moment_condition = {
      function_value: parameters -> float -> float;
      target_value: float;
      weight: float;
    }

    val estimate_moments :
      moment_condition array ->
      parameters ->
      float array ->
      estimation_config -> parameters
  end

  module MCMCEstimation : sig
    type mcmc_config = {
      n_chains: int;
      burnin: int;
      thin: int;
      proposal_std: float;
    }

    val metropolis_hastings :
      log_likelihood:(parameters -> float array -> float) ->
      prior:(parameters -> float) ->
      config:mcmc_config ->
      float array -> parameters array array
  end

  module GMMEstimation : sig
    val two_step_gmm :
      (parameters -> float -> float) array ->
      (float -> float) array ->
      parameters ->
      float array ->
      estimation_config -> parameters
  end
end

(* Numerical Methods *)
module NumericalMethods : sig
  val euler_maruyama :
    drift:(float -> float) ->
    diffusion:(float -> float) ->
    init:float ->
    dt:float ->
    steps:int -> float array

  val milstein :
    drift:(float -> float) ->
    diffusion:(float -> float) ->
    diffusion_derivative:(float -> float) ->
    init:float ->
    dt:float ->
    steps:int -> float array

  val taylor_1_5 :
    drift:(float -> float) ->
    diffusion:(float -> float) ->
    drift_derivative:(float -> float) ->
    diffusion_derivative:(float -> float) ->
    mixed_derivative:(float -> float) ->
    init:float ->
    dt:float ->
    steps:int -> float array

  val adaptive_step :
    drift:(float -> float) ->
    diffusion:(float -> float) ->
    init:float ->
    dt:float ->
    tol:float ->
    max_steps:int -> float list
end