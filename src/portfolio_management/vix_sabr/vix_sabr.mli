open Torch

(** Core types *)
type parameters = {
  beta: float;
  omega: float;  (** volatility of volatility *)
  rho: float;    (** correlation *)
  alpha: float;  (** = (1-beta)/2 *)
  delta: float;  (** = -(1-beta)ρω *)
}

type state = {
  s: float Tensor.t;  (** Asset price *)
  v: float Tensor.t;  (** Volatility *)
  t: float;          (** Current time *)
}

type simulation_config = {
  n_paths: int;
  n_steps: int;
  dt: float;
  use_brownian_bridge: bool;
  variance_reduction: bool;
  antithetic: bool;
}

type integration_config = {
  rel_tol: float;
  abs_tol: float;
  max_iter: int;
  method_name: string;
}

type boundary_type =
  | Regular     (** Accessible and can exit *)
  | Exit       (** Accessible but cannot return *)
  | Entrance   (** Inaccessible but can enter *)
  | Natural    (** Inaccessible and cannot enter *)

type boundary_classification = {
  zero_type: boundary_type;
  infinity_type: boundary_type;
}

type solver_config = {
  rel_tol: float;
  abs_tol: float;
  max_steps: int;
  stability_factor: float;
}

type 'a integration_result = {
  value: 'a;
  error_est: float;
  n_steps: int;
}

type time_span = {
  t0: float;
  t1: float;
}

(** Integration function *)
val integrate : ?config:integration_config -> (float -> float) -> float -> float -> 
               (float integration_result, string) result

(** SABR model *)
module Sabr : sig
  val create_parameters : beta:float -> omega:float -> rho:float -> parameters
  val sigma_v : parameters -> float Tensor.t -> float Tensor.t
  val mu_v : parameters -> float Tensor.t -> float Tensor.t
  val evolve_step : parameters -> state -> float -> state
  val simulate : parameters -> state -> simulation_config -> state list
  val validate_parameters : parameters -> (parameters, string) result
end

(** Natural scale calculations *)
val calc_R : parameters -> float Tensor.t -> float Tensor.t
val calc_F : parameters -> float Tensor.t -> float Tensor.t
val calc_scale_function : parameters -> float -> float

(** Stochastic integration *)
module Milstein : sig
  type scheme_config = {
    use_antithetic: bool;
    correction_term: bool;
    n_paths: int;
    dt: float;
  }

  val calc_derivatives : parameters -> float Tensor.t -> float Tensor.t * float Tensor.t
  val step : parameters -> state -> scheme_config -> float -> state
  val simulate : parameters -> state -> scheme_config -> int -> state list
end

module WagnerPlaten : sig
  val step : parameters -> state -> float -> state
  val simulate : parameters -> state -> float -> int -> state list
end

(** VIX calculation and pricing *)
module Vix : sig
  val calc_vix : parameters -> state -> float -> float

  module Options : sig
    type pricing_config = {
      n_paths: int;
      dt: float;
      variance_reduction: bool;
      error_est: bool;
    }

    type pricing_result = {
      price: float;
      error: float option;
      greeks: greeks option;
      implied_vol: float option;
    }
    and greeks = {
      delta: float;
      gamma: float;
      vega: float;
      theta: float;
      rho: float;
    }

    val black_scholes : s:float -> k:float -> r:float -> t:float -> sigma:float -> 
                       (float * float) * greeks

    val price_call : parameters -> state -> strike:float -> maturity:float -> 
                    pricing_config -> float * pricing_result

    val price_put : parameters -> state -> strike:float -> maturity:float -> 
                   pricing_config -> float * pricing_result
  end

  module Futures : sig
    val price : parameters -> state -> float -> Vix.Options.pricing_config -> 
                float * float option
  end
end

(** Capped volatility process *)
module CappedVolatility : sig
  type cap_parameters = {
    a: float;  (** volatility cap *)
    b: float;  (** drift cap *)
    base_params: parameters;
  }

  val capped_sigma_v : cap_parameters -> float Tensor.t -> float Tensor.t
  val capped_mu_v : cap_parameters -> float Tensor.t -> float Tensor.t
  val compute_v_hat : cap_parameters -> float
  val step : cap_parameters -> state -> float -> state
  val calc_vix : cap_parameters -> state -> float -> float

  module Options : sig
    type pricing_result = {
      price: float;
      error: float option;
      greeks: Vix.Options.greeks option;
      implied_vol: float option;
    }

    val price_call : cap_parameters -> state -> strike:float -> maturity:float -> 
                    config:Vix.Options.pricing_config -> float * pricing_result

    val price_put : cap_parameters -> state -> strike:float -> maturity:float -> 
                   config:Vix.Options.pricing_config -> float * pricing_result
  end
end

(** Short maturity asymptotics *)
module ShortMaturity : sig
  val compute_rate_function : parameters -> float -> float -> float
  val asymptotic_prices : parameters -> float -> float -> float -> float

  module ImpliedVol : sig
    val atm_level : parameters -> float -> float
    val skew : parameters -> float -> float
    val convexity : parameters -> float -> float
  end

  val implied_vol_expansion : parameters -> float -> float -> float -> float
end

(** Model calibration *)
module Calibration : sig
  type market_data = {
    strikes: float array;
    maturities: float array;
    call_prices: float array array;
    put_prices: float array array;
    call_ivols: float array array option;
    put_ivols: float array array option;
  }

  type calibration_config = {
    max_iter: int;
    tolerance: float;
    regularization: float;
    method_name: string;
  }

  type calibration_result = {
    parameters: parameters;
    error: float;
    n_iterations: int;
    convergence: bool;
    fit_quality: fit_metrics;
  }
  and fit_metrics = {
    rmse: float;
    max_error: float;
    avg_error: float;
    r_squared: float;
  }

  val calculate_loss : parameters -> market_data -> calibration_config -> float
  val enforce_constraints : parameters -> parameters
  val optimize : parameters -> market_data -> calibration_config -> 
                (calibration_result, string) result

  module Analysis : sig
    val compute_vol_surface : parameters -> strikes:float array -> 
                            maturities:float array -> float array array
    val analyze_stability : parameters -> market_data -> n_trials:int -> 
                          {
                            success_rate: float;
                            parameter_stability: {
                              beta: float * float;
                              omega: float * float;
                              rho: float * float;
                            }
                          }
    val analyze_fit_quality : parameters -> market_data -> 
                            {
                              strike_slice_errors: (float * float) array;
                              maturity_slice_errors: (float * float) array;
                              total_rmse: fit_metrics;
                            }
  end
end