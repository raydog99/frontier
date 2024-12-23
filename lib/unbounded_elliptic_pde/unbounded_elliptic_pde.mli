open Torch

val pi : float

(** L2 norm of a tensor *)
val l2_norm : Tensor.t -> Tensor.t

(** Compute gradient using finite differences *)
val gradient : (Tensor.t -> Tensor.t) -> Tensor.t -> float -> Tensor.t

(** Compute Laplacian using finite differences *)
val laplacian : (Tensor.t -> Tensor.t) -> Tensor.t -> float -> float

module Domain : sig
  type t
  
  val create : dim:int -> size:float -> t
  val get_subdomain : t -> float -> t
  val get_boundary_layer : t -> t
  val to_indices : t -> Tensor.t -> Tensor.t
  
  val size : t -> float
  val dim : t -> int
  val grid_size : t -> int
  val points : t -> Tensor.t array
end

module Coefficient : sig
  type t
  
  val create : (Tensor.t -> Tensor.t) -> t
  val create_periodic : base_func:(Tensor.t -> Tensor.t) -> 
                       period:float array -> dim:int -> t
  val create_quasiperiodic : scale:float -> t
  
  val evaluate : t -> Tensor.t -> Tensor.t
  val check_bounds : t -> alpha:float -> beta:float -> bool
end

module Regularization : sig
  type t = {
    time_param: float;
    exp_order: int;
    boundary_param: float;
  }
  
  val create : time_param:float -> exp_order:int -> boundary_param:float -> t
  val optimal_params : Domain.t -> float -> t
end

module Solver : sig
  type t
  
  val create : Domain.t -> Coefficient.t -> t
  val with_regularization : t -> Regularization.t -> t
  
  val solve_standard : t -> Tensor.t -> Tensor.t
  val solve_regularized : t -> Tensor.t -> Tensor.t
  val solve_spectral : t -> Tensor.t -> Tensor.t
  
  val compute_l2_error : t -> Domain.t -> Tensor.t -> Tensor.t -> float
end

module ErrorAnalysis : sig
  type error_components = {
    boundary_error: float;
    modeling_error: float;
    total_error: float;
  }
  
  val analyze_error : Solver.t -> Tensor.t -> error_components
  val estimate_convergence_rate : float array -> float array -> float
  val verify_exponential_decay : Solver.t -> Tensor.t -> bool
end

module SpectralAnalysis : sig
  type spectrum = {
    frequencies: float array;
    amplitudes: float array;
    decay_rate: float option;
  }
  
  val compute_spectrum : Tensor.t -> spectrum
  val analyze_periodicity : Coefficient.t -> Domain.t -> float array option
  val estimate_decay_rate : spectrum -> float
end

module Discretization : sig
  type grid = {
    points: Tensor.t array;
    weights: Tensor.t;
  }
  
  val create_grid : Domain.t -> grid
  val interpolate : grid -> Tensor.t -> Tensor.t -> Tensor.t
end

module MatrixOperations : sig
  type operator = {
    size: int;
    apply: Tensor.t -> Tensor.t;
  }
  
  val apply_matrix_exponential : operator -> Tensor.t -> float -> Tensor.t
  val solve_linear_system : operator -> Tensor.t -> Tensor.t
end