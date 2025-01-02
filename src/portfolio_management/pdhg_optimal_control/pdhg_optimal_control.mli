open Torch

type dim1 = {
  nx: int;
  nt: int;
}

type dim2 = {
  nx: int;
  ny: int;
  nt: int;
}

type config = {
  dt: float;
  dx: float;
  dy: float option;
  epsilon: float;
  tau_rho: float;
  tau_alpha: float;
  tau_phi: float;
  max_iter: int;
  tolerance: float;
}

val d_plus_x : Tensor.t -> float -> Tensor.t
val d_minus_x : Tensor.t -> float -> Tensor.t
val laplacian : Tensor.t -> float -> Tensor.t
val d_time : Tensor.t -> float -> Tensor.t
val gradient : Tensor.t -> Tensor.t
val adjoint : Tensor.t -> Tensor.t
val second_derivative_x : Tensor.t -> float -> Tensor.t
val second_derivative_y : Tensor.t -> float -> Tensor.t
val cross_derivative_xy : Tensor.t -> float -> float -> Tensor.t

module Domain : sig
  type t = {
    x_min: float;
    x_max: float;
    y_min: float option;
    y_max: float option;
    t_min: float;
    t_max: float;
  }

  val create_1d : x_min:float -> x_max:float -> t_min:float -> t_max:float -> t
  val create_2d : x_min:float -> x_max:float -> y_min:float -> y_max:float -> 
                  t_min:float -> t_max:float -> t
end

module OptimalControl : sig
  type state = {
    phi: Tensor.t;
    rho: Tensor.t;
    alpha: Tensor.t;
    phi_tilde: Tensor.t;
  }

  val create_state : nx:int -> nt:int -> initial:Tensor.t -> state

  module PDHG : sig
    val step : config -> state -> state
    val solve : ?max_iter:int -> ?tol:float -> config -> state -> state
  end
end

module CompleteSaddlePoint : sig
  type density_state = {
    rho: Tensor.t;
    active_set: Tensor.t;
    inactive_set: Tensor.t;
  }

  val update_density_state : rho:Tensor.t -> epsilon:float -> density_state
  val modified_pdhg_step : state:density_state -> config:config -> Tensor.t
end

module CompletePhaseSpace : sig
  type phase_state = {
    position: Tensor.t;
    velocity: Tensor.t;
    momentum: Tensor.t;
    time: float;
  }

  val symplectic_evolve : state:phase_state -> dt:float -> 
                         force:(Tensor.t -> Tensor.t -> float -> Tensor.t) -> phase_state
  val solve : initial:phase_state -> config:config -> phase_state
end

module CompleteFokkerPlanck : sig
  type fp_state = {
    density: Tensor.t;
    drift: Tensor.t;
    diffusion: Tensor.t;
    time: float;
  }

  val solve_fokker_planck : initial:Tensor.t -> 
                           drift:(Tensor.t -> float -> Tensor.t) ->
                           diffusion:(Tensor.t -> float -> Tensor.t) ->
                           config:config -> fp_state
end

module CompleteMixedBoundary : sig
  type boundary_type =
    | Periodic
    | Neumann
    | Dirichlet of float
    | Mixed of boundary_type * boundary_type

  val apply_mixed_boundary : state:OptimalControl.state -> 
                           bc_type:boundary_type ->
                           config:config -> OptimalControl.state
end

module NewtonMechanics : sig
  type state = {
    position: Tensor.t;
    velocity: Tensor.t;
    momentum: Tensor.t;
    time: float;
  }

  val create_state : initial_pos:Tensor.t -> initial_vel:Tensor.t -> time:float -> state
  val solve_newton_mechanics : initial:state -> force:Tensor.t -> config:config -> state
end

module StochasticControl : sig
  type stochastic_state = {
    value: Tensor.t;
    noise: Tensor.t;
    paths: Tensor.t list;
    time: float;
  }

  val create_stochastic_state : initial:Tensor.t -> time:float -> stochastic_state
  val solve_viscous_hjb : hamiltonian:Tensor.t -> epsilon:float -> 
                         config:config -> initial:stochastic_state -> stochastic_state
end

module TimeDependent : sig
  type coefficient = {
    spatial: Tensor.t -> float -> Tensor.t;
    temporal: float -> Tensor.t -> Tensor.t;
    mixed: Tensor.t -> float -> Tensor.t;
  }

  val eval_coefficients : x:Tensor.t -> t:float -> coefficient -> 
    {space_coeff: Tensor.t; time_coeff: Tensor.t; mixed_coeff: Tensor.t}
end

module ViscousTreatment : sig
  type viscous_params = {
    epsilon: float;
    anisotropic: bool;
    dimension_coeffs: float array;
  }

  val discretize_viscous_terms : phi:Tensor.t -> params:viscous_params -> 
                                config:config -> Tensor.t
  val apply_viscous_boundary : state:OptimalControl.state -> params:viscous_params -> 
                             config:config -> OptimalControl.state
end