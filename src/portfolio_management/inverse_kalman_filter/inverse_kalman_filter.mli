open Torch

module Types : sig
  type state = {
    mean: Tensor.t;
    covariance: Tensor.t;
  }

  type observation = {
    value: Tensor.t;
    variance: Tensor.t;
  }

  type dlm_params = {
    ft: Tensor.t;  (* Observation matrix *)
    gt: Tensor.t;  (* State transition matrix *)
    wt: Tensor.t;  (* Process noise covariance *)
    vt: Tensor.t;  (* Observation noise variance *)
    bt: Tensor.t;  (* State prediction covariance *)
    qt: Tensor.t;  (* Observation prediction variance *)
    kt: Tensor.t;  (* Kalman gain *)
  }

  type model_params = {
    gamma: Tensor.t;  (* Range parameters *)
    eta: Tensor.t;    (* Variance ratios *)
    sigma2_0: float;  (* Noise variance *)
  }

  type training_data = {
    inputs: Tensor.t;
    outputs: Tensor.t;
    num_states: int;
    num_obs: int;
  }

  type evaluation_metrics = {
    nrmse: float;
    credible_interval_length: float;
    coverage_proportion: float;
  }
end

module KalmanFilter : sig  
  val compute_filter_matrices : 
    Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> int -> int -> Types.dlm_params list
end

module InverseKalmanFilter : sig
  val compute_lt_transpose_u : Types.dlm_params list -> Tensor.t -> Tensor.t
  val compute_l_x_tilde : Types.dlm_params list -> Tensor.t -> Tensor.t
  val compute_sigma_u : Types.dlm_params list -> Tensor.t -> Tensor.t
  val compute_sigma_u_robust : Types.dlm_params list -> Tensor.t -> float -> Tensor.t
  val compute_lt_inv_x_tilde : Types.dlm_params list -> Tensor.t -> Tensor.t
end

module ConjugateGradient : sig
  val solve : (Tensor.t -> Tensor.t) -> Tensor.t -> int -> float -> Tensor.t
end

module IkfCg : sig
  val matvec : Types.dlm_params list -> Tensor.t -> Tensor.t
  val predict_mean : Types.dlm_params list -> Types.observation -> Tensor.t
  val predict_variance : Types.dlm_params list -> Tensor.t -> Tensor.t
end

module ParameterEstimation : sig
  val approximate_log_det : Tensor.t -> int -> int -> Tensor.t
  val cross_validate : Types.model_params -> Types.training_data -> Types.training_data -> Tensor.t
  val maximum_likelihood : Types.model_params -> Types.training_data -> Tensor.t
end

module MaternKernel : sig
  val matern_to_dlm : float -> float -> float -> Types.dlm_params
end

module Evaluation : sig
  val compute_nrmse : Tensor.t -> Tensor.t -> float
  val compute_interval_length : Tensor.t -> float
  val compute_coverage : Tensor.t -> Tensor.t -> Tensor.t -> float
  val evaluate : Tensor.t -> Tensor.t -> Tensor.t -> Types.evaluation_metrics
end