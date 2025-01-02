open Torch

val pi : float
val gamma : float -> float
val fourier_transform : (float array -> float) -> int -> float array -> float
val bessel_j : float -> float -> float
val bessel_k : float -> float -> float
val dct1 : float array -> float array
val dst1 : float array -> float array
val ks_test : float array -> (float -> float) -> bool
val chi_square_test : float array -> int -> (float -> float) -> bool
val anderson_darling_test : float array -> (float -> float) -> bool
val covariance_test : 
  (float array -> float) -> float array array -> int -> bool

module MaternCovariance : sig
  type t = {
    nu: float;
    ell: float;
    sigma: float;
  }
  val create : nu:float -> ell:float -> sigma:float -> t
  val eval : t -> float array -> float -> float
  val fourier_transform : t -> float array -> float -> float
end

module Lattice : sig
  type multiindex = int array
  type hyperoctant = {
    q: int array;
    dim: int;
  }
  val create_hyperoctant : int -> int array -> hyperoctant
  val in_hyperoctant : hyperoctant -> multiindex -> bool
  val truncated_lattice : int -> int -> multiindex list
end

module FunctionSpace : sig
  type function_type = Continuous | L2 | H1 | Sobolev of int
  type domain = {
    dim: int;
    bounds: (float * float) array;
  }
  val create_domain : int -> (float * float) array -> domain
  val norm : function_type -> (float array -> float) -> domain -> float
end

module GRF : sig
  type t = {
    covar: MaternCovariance.t;
    alpha: float;
    domain_dim: int;
  }
  val create : covar:MaternCovariance.t -> alpha:float -> domain_dim:int -> t
  val sample : t -> int -> float array array -> float array
  val compute_dna_coeffs : t -> int -> (int array * float) list
  val enhanced_sample : t -> int -> float array array -> float array
end

module FEM : sig
  type element = {
    nodes: float array array;
    basis: float array;
    weights: float array;
  }
  type mesh = {
    elements: element array;
    h: float;
  }
  val create_uniform_mesh : int -> int -> mesh
  val assemble_stiffness : mesh -> SPDE.params -> float array array
end

module SPDE : sig
  type params = {
    kappa: float;
    beta: float;
    dim: int;
  }
  val create : nu:float -> dim:int -> params
  val apply_operator : params -> (float array -> float) -> float array -> float
  val white_noise : int -> int -> float array array
end

val spde_test : 
  SPDE.params -> float array -> FEM.mesh -> bool

module BoundaryConditions : sig
  type bc_type =
    | Dirichlet of float
    | Neumann of float
    | Robin of float * float
    | Periodic
    | Mixed of bc_type array

  type boundary = {
    bc_type: bc_type;
    boundary_index: int;
    dimension: int;
  }

  val apply_bc : FEM.mesh -> boundary -> float array array -> float array -> unit
  val verify_bc : float array -> boundary -> float -> bool
end

module DNAIntegration : sig
  type boundary_config = {
    dirichlet_weight: float;
    neumann_weight: float;
    boundary_conditions: BoundaryConditions.boundary array;
  }

  val sample_with_boundaries : GRF.t -> boundary_config -> 
    float array array -> float array
  val verify_equivalence : GRF.t -> SPDE.params -> 
    float array array -> (bc_type * float) list
end