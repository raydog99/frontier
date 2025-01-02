open Torch

module type SABR = sig
  type t
  type model = Standard | Shifted of float | Dynamic of (float -> float)

  val create : float -> float -> float -> float -> float -> model -> t
  val simulate : t -> float -> int -> Tensor.t * Tensor.t
  val simulate_terminal : t -> float -> int -> Tensor.t
  val price_european_options : t -> float -> float array -> int -> [`Call | `Put] -> (float * float) array
  val implied_volatilities : t -> float -> float array -> float array -> [`Call | `Put] -> (float * float) array
  val hagan_formula : t -> float -> float -> float
  val calibrate : float -> float array -> float array -> float -> [`Call | `Put] -> t
  val local_volatility : t -> float -> float -> float
  val forward_volatility : t -> float -> float -> float -> float
  val delta : t -> float -> float -> float -> [`Call | `Put] -> float
  val vega : t -> float -> float -> float -> [`Call | `Put] -> float
  val rho_sensitivity : t -> float -> float -> float -> [`Call | `Put] -> float
end

module Sabr : SABR

module Utils : sig
  val normal_cdf : float -> float
  val normal_pdf : float -> float
  val sample_gamma : float -> float -> float
  val sample_poisson : float -> int
  val black_scholes_price : float -> float -> float -> float -> float -> float
  val black_scholes_implied_vol : float -> float -> float -> float -> float -> float
  val parallel_map : ('a -> 'b) -> 'a array -> 'b array
  val cubic_spline_interpolation : float array -> float array -> (float -> float)
end

module CEV : sig
  val sample : float -> float -> float -> float -> float
end

module NumericalIntegration : sig
  val gauss_legendre : (float -> float) -> float -> float -> int -> float
  val adaptive_simpson : (float -> float) -> float -> float -> float -> float
end

exception Invalid_parameter of string
exception Calibration_error of string

module Log : sig
  val info : string -> unit
  val warn : string -> unit
  val error : string -> unit
end

module Test : sig
  val run_all_tests : unit -> unit
end