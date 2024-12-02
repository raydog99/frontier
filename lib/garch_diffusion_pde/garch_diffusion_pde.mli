open Torch

type model_params = {
  v0: float;         (** Initial variance *)
  vbar: float;      (** Long-run mean variance *)
  kappa: float;     (** Mean reversion rate *)
  xi: float;        (** Volatility of variance *)
  rho: float;       (** Correlation coefficient *)
}

type grid_params = {
  s_min: float;
  s_max: float;
  v_min: float;
  v_max: float;
  ns: int;          (** Number of S grid points *)
  nv: int;          (** Number of V grid points *)
  nt: int;          (** Number of time steps *)
  dt: float;        (** Time step size *)
}

type solver_type = [ `ADI_HV | `ADI_MCS | `BDF3 ]

type option_data = {
  strike: float;
  expiry: float;
  market_price: float;
  option_type: [ `Call | `Put ];
}

module GridGen : sig
  val gen_s_grid : grid_params -> strike:float -> Tensor.t
  val gen_v_grid : grid_params -> v0:float -> Tensor.t
  val calc_bounds : model_params -> option_data -> float * float * float * float
end

module FiniteDiff : sig
  val first_deriv_coeff : float -> float -> float * float * float
  val second_deriv_coeff : float -> float -> float * float * float
  val mixed_deriv_coeff : float -> float -> float -> float -> float
end

module ADI : sig
  type operator_split = {
    a0: Tensor.t;  (** Mixed derivative terms *)
    a1: Tensor.t;  (** S direction terms *)
    a2: Tensor.t;  (** V direction terms *)
    b: Tensor.t;   (** Source/reaction terms *)
  }

  module HV : sig
    val step : dt:float -> operator_split -> Tensor.t -> Tensor.t
  end

  module MCS : sig
    val step : dt:float -> operator_split -> Tensor.t -> Tensor.t
  end
end

module BDF3 : sig
  val solve : model_params -> ADI.operator_split -> Tensor.t -> Tensor.t
  val implicit_euler_step : dt:float -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
  val bdf2_step : dt:float -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
end

val build_coefficient_matrix : model_params -> grid_params -> 
  Tensor.t -> Tensor.t -> ADI.operator_split

module Richardson : sig
  type extrap_config = {
    space_ratio: float;
    time_ratio: float;
    space_order: int;
    time_order: int;
  }

  val extrapolate : (model_params -> Tensor.t) -> model_params -> extrap_config -> 
    Tensor.t * float
end

module Calibration : sig
  type calibration_mode = Price_Based | IV_Based | Hybrid
  
  type calibration_result = {
    fitted_params: model_params;
    rmse: float;
    iterations: int;
    time_taken: float;
    stability_metric: float;
    negative_values: int;
    parameter_path: model_params list;
  }

  val calibrate : model_params -> option_data list -> calibration_mode -> 
    (calibration_result, string) result
end

module ModelComparison : sig
  type model = 
    | GARCH of model_params
    | Heston of model_params
    | PModel of {base: model_params; p: float}

  type comparison_metrics = {
    rmse_iv: float;
    avg_calib_time: float;
    param_stability: float;
    short_term_fit: float;
    long_term_fit: float;
    smile_coverage: float;
    feller_ratio: float option;
  }

  val compare_models : model list -> option_data list -> (model * comparison_metrics) list
end