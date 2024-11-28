open Torch

type t

val integrate : f:(float -> float) -> a:float -> b:float -> n:int -> float
val derivative : f:(float -> float) -> x:float -> h:float -> float
val second_derivative : f:(float -> float) -> x:float -> h:float -> float
val solve_tridiagonal : 
	float array -> float array -> float array -> float array -> float array d

val create : volatility:(float -> float) -> brownian:BrownianMotion.t -> t
val simulate : t -> initial_value:float -> Tensor.t
val quadratic_normal : alpha0:float -> alpha1:float -> alpha2:float -> float -> float
val bessel_2d : float -> float

module Scale : sig
  type t = private {
    volatility: float -> float;
    phi_up: float -> float;
    phi_down: float -> float;
    lambda: float
  }

  val create : volatility:(float -> float) -> lambda:float -> t
  val phi_combined : t -> float -> float
end

module BrownianMotion : sig
  type t

  val create : dt:float -> num_steps:int -> device:Device.t -> t
  val simulate : t -> initial_value:float -> Tensor.t
end

module HarmonicFunction : sig
  type t

  val create : lambda:float -> scale:(float -> float) -> volatility:(float -> float) -> t
  val evaluate : t -> float -> float
  val is_strictly_convex : t -> float -> float -> float -> bool
end

module PDE : sig
  type solution = {
    value: float -> float -> float;
    gradient: float -> float -> float;
  }

  val solve_weak_pde : 
    volatility:(float -> float) ->
    initial_condition:(float -> float) ->
    lambda:float ->
    t_max:float ->
    x_grid:float array ->
    solution
end

module HigherMoments : sig
  type moment_bound = {
    p: float;
    c: float;
  }

  module RegularizedMartingale : sig
    type t

    val create : 
      process:StrictLocalMartingale.t -> 
      T:float -> 
      h_star:(float -> float -> float) -> t
    val simulate : t -> initial_value:float -> num_steps:int -> Tensor.t
  end

  val check_polynomial_growth : bound:moment_bound -> f:(float -> float) -> float -> bool
end

module InverseBessel3D : sig
  type t

  val create : initial_value:float -> t
  val simulate : t -> dt:float -> num_steps:int -> Tensor.t
  val second_moment : t -> float -> float
  val quadratic_variation : t -> float -> float
end

module MartingaleMethods : sig
  type martingale_type = TrueMartingale | StrictLocalMartingale | Submartingale | Unknown

  val classify_process :
    volatility:(float -> float) ->
    initial_value:float ->
    t_max:float ->
    martingale_type
end

module NumericalMethods : sig
  module AdaptiveSteps : sig
    type error_control = {
      abs_tol: float;
      rel_tol: float;
      safety_factor: float;
    }

    val runge_kutta_adaptive :
      system:(float -> float -> float) ->
      initial:float ->
      t_span:(float * float) ->
      control:error_control ->
      float list * float list
  end

  module SpectralMethods : sig
    val chebyshev_diff_matrix : int -> float array array
    val solve_spectral_pde :
      pde_func:(float -> float -> float -> float -> float) ->
      initial_condition:(float -> float) ->
      boundary_conditions:(float -> float) * (float -> float) ->
      grid:PDE.solution ->
      float array array
  end
end