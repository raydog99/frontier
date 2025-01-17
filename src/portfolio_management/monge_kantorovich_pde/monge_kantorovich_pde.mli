open Torch

type measure = {
  density: Tensor.t;
  support: Tensor.t;
}

module Divergence : sig
  type t = Tensor.t -> Tensor.t -> Tensor.t
  
  val p_wasserstein : int -> t
  val radial_divergence : (Tensor.t -> Tensor.t) -> t
  val heat_coupling_divergence : float -> t
  val variable_coeff_divergence : (Tensor.t -> Tensor.t) -> t
end

module Measure : sig
  type t = measure

  val create : Tensor.t -> Tensor.t -> t
  val marginal_x : measure -> measure
  val marginal_y : measure -> measure
  val is_probability_measure : measure -> bool
  val compute_marginals : Tensor.t -> int list -> Tensor.t * Tensor.t
  val project_probability_measure : measure -> measure
end

module MongeKantorovich : sig
  type coupling = {
    joint_density: Tensor.t;
    marginal_x: measure;
    marginal_y: measure;
  }

  val create_coupling : measure -> measure -> coupling
  val divergence : Divergence.t -> measure -> measure -> Tensor.t
  val sinkhorn_divergence : float -> Tensor.t -> Tensor.t -> Tensor.t -> int -> 
    Tensor.t * Tensor.t
end

module TimeIntegrator : sig
  type t = ForwardEuler | RK4 | TVD_RK3

  val step : t -> dt:float -> (float -> Tensor.t -> Tensor.t) -> 
    float -> Tensor.t -> Tensor.t
end

module Grid : sig
  val uniform_grid : min:float -> max:float -> points:int -> Tensor.t
  val derivative_1d : input:Tensor.t -> dx:float -> Tensor.t
  val laplacian_1d : input:Tensor.t -> dx:float -> Tensor.t
  val weno5_reconstruction : Tensor.t -> Tensor.t

  type boundary_condition =
    | Dirichlet of float
    | Neumann of float
    | Periodic
    | Robin of float * float

  val apply_boundary_condition : boundary_condition -> Tensor.t -> float -> unit
end

module NumericalMethods : sig
  val tvd_limit : Tensor.t -> Tensor.t
  val entropy_stable_flux : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
  val compute_timestep : dx:float -> cfl:float -> max_speed:float -> float
  val estimate_error : Tensor.t -> float -> float -> float * float
end

module PDE : sig
  type boundary_type = Grid.boundary_condition
  type base_params = {
    dx: float;
    dt: float;
    bc_left: boundary_type;
    bc_right: boundary_type;
  }
end

module HeatEquation : sig
  type params = {
    base: PDE.base_params;
    diffusion_coeff: float;
  }

  val rhs : params -> Tensor.t -> Tensor.t
  val solve : params -> Tensor.t -> float -> Tensor.t
end

module VariableCoefficientHeat : sig
  type coeff_params = {
    base: PDE.base_params;
    a: Tensor.t -> Tensor.t;
    da_dx: Tensor.t -> Tensor.t;
  }

  val flux_discretization : coeff_params -> Tensor.t -> Tensor.t
  val solve : coeff_params -> Tensor.t -> float -> Tensor.t
end

module FokkerPlanck : sig
  type vector_field = {
    drift: Tensor.t -> float -> Tensor.t;
    divergence: Tensor.t -> float -> Tensor.t;
    jacobian: Tensor.t -> float -> Tensor.t;
  }

  type params = {
    base: PDE.base_params;
    vector_field: vector_field;
    diffusion: Tensor.t;
  }

  val spatial_discretization : params -> Tensor.t -> float -> Tensor.t
  val solve : params -> Tensor.t -> float -> Tensor.t
end

module Scattering : sig
  type kernel = {
    phi: Tensor.t -> Tensor.t -> Tensor.t;
    dphi_dv: Tensor.t -> Tensor.t -> Tensor.t;
    mu: float -> float;
  }

  type params = {
    base: PDE.base_params;
    kernel: kernel;
  }

  val compute_integral : kernel -> Tensor.t -> Tensor.t -> Tensor.t
  val solve : params -> Tensor.t -> Tensor.t -> float -> Tensor.t
end

module Boltzmann : sig
  type collision_kernel = {
    b_theta: float -> float;
    v_to_v_star: Tensor.t -> Tensor.t -> float -> float -> Tensor.t * Tensor.t;
  }

  type params = {
    base: PDE.base_params;
    kernel: collision_kernel;
  }

  val maxwell_post_collision : Tensor.t -> Tensor.t -> float -> float -> 
    Tensor.t * Tensor.t
  val maxwell_kernel : unit -> collision_kernel
  val compute_collision_integral : params -> Tensor.t -> Tensor.t -> Tensor.t
  val solve : params -> Tensor.t -> Tensor.t -> float -> Tensor.t
end

module Optimization : sig
  val inplace_operation : (Tensor.t -> Tensor.t -> Tensor.t) -> 
    Tensor.t -> Tensor.t -> unit
  val memoize : ('a -> 'b) -> 'a -> 'b
  val adaptive_timestep : dt:float -> cfl:float -> max_velocity:float -> float
  val parallel_collision_integral : Boltzmann.params -> Tensor.t -> Tensor.t -> 
    int -> Tensor.t
end

module Stability : sig
  val tvd_entropy_limit : Tensor.t -> float -> Tensor.t
  val flux_correction : Tensor.t -> Tensor.t -> Tensor.t
end

module Integration : sig
  type equation_type =
    | Heat of VariableCoefficientHeat.coeff_params
    | FokkerPlanck of FokkerPlanck.params
    | Scattering of Scattering.params
    | Boltzmann of Boltzmann.params

  val solve : equation_type -> Tensor.t -> Tensor.t list -> float -> 
    adaptive:bool -> Tensor.t
  val compute_error : Tensor.t -> Tensor.t -> float -> float -> 
    float * float * float
  val analyze_solution : Tensor.t -> Tensor.t * Tensor.t * Tensor.t * Tensor.t
  val generate_report : equation_type -> Tensor.t -> 
    (float * float * float) -> 
    (Tensor.t * Tensor.t * (Tensor.t * Tensor.t)) -> string
end