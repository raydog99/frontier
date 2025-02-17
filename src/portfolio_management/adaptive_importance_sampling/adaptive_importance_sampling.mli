open Torch

module type Process = sig
  type t
  val create : unit -> t
  val step : t -> Tensor.t -> Tensor.t
  val get_state : t -> Tensor.t
  val set_state : t -> Tensor.t -> unit
end

module ForwardSDE : sig
  type t
  
  val create : Tensor.t -> float -> (Tensor.t -> float -> Tensor.t) -> (Tensor.t -> Tensor.t) -> t
  val get_state : t -> Tensor.t
  val set_state : t -> Tensor.t -> unit
  val get_time : t -> float
  val step : t -> Tensor.t -> Tensor.t
end

module BackwardSDE : sig
  type t
  
  val create : Tensor.t -> Tensor.t -> float -> t
  val get_y : t -> Tensor.t
  val get_z : t -> Tensor.t
  val set_y : t -> Tensor.t -> unit
  val set_z : t -> Tensor.t -> unit
  val step : t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
end

module NumericalSchemes : sig
  val euler_maruyama_step : Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t -> Tensor.t
  val backward_euler_step : Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t -> Tensor.t
  val generate_increments : int -> int -> float -> Tensor.t
end

module BasisFunctions : sig
  type t = {
    functions: (Tensor.t -> Tensor.t) array;
    gradients: (Tensor.t -> Tensor.t) array;
  }
  
  val create : (Tensor.t -> Tensor.t) array -> (Tensor.t -> Tensor.t) array -> t
  val evaluate : t -> Tensor.t -> int -> Tensor.t
  val evaluate_gradient : t -> Tensor.t -> int -> Tensor.t
  val create_rbf_basis : float array -> float -> t
end

module type MeasureSpace = sig
  type t
  type path
  type functional
  
  val create : dim:int -> t
  val density : t -> path -> Tensor.t
  val expectation : t -> functional -> Tensor.t
  val relative_entropy : t -> t -> Tensor.t
  val is_absolutely_continuous : t -> t -> bool
end

module PathMeasure : sig
  type t = {
    drift: Tensor.t -> float -> Tensor.t;
    diffusion: Tensor.t -> Tensor.t;
    dim: int;
    time_step: float;
  }
  
  val create : (Tensor.t -> float -> Tensor.t) -> (Tensor.t -> Tensor.t) -> int -> float -> t
  val transform_measure : t -> (Tensor.t -> float -> Tensor.t) -> t
  val check_novikov : t -> (Tensor.t -> float -> Tensor.t) -> Tensor.t array -> bool
end

module ImportanceSampling : sig
  type measure = {
    drift: Tensor.t -> float -> Tensor.t;
    diffusion: Tensor.t -> Tensor.t;
    dimension: int;
  }
  
  type t
  
  val create : measure -> measure -> float -> float -> t
  val log_likelihood_ratio : t -> Tensor.t array -> Tensor.t
  val zero_variance_estimator : t -> Tensor.t array array -> (Tensor.t array -> Tensor.t) -> Tensor.t
end

module TrajectoryManager : sig
  type path = {
    states: Tensor.t array;
    times: float array;
    terminal_idx: int option;
  }
  
  type t
  
  val create : int -> int -> float array -> t
  val get_state : t -> int -> int -> Tensor.t
  val set_state : t -> int -> int -> Tensor.t -> unit
  val set_terminal : t -> int -> int -> unit
  val is_terminated : t -> int -> int -> bool
end

module LeastSquaresSolvers : sig
  type solver_type = SVD | QR | Cholesky
  type t
  
  val create : solver_type -> int -> float -> t
  val solve : t -> Tensor.t -> Tensor.t -> Tensor.t
end

module AdvancedLSMC : sig
  type config = {
    num_paths: int;
    time_steps: int;
    basis_size: int;
    solver: LeastSquaresSolvers.t;
  }
  
  type t
  
  val create : config -> BasisFunctions.t -> TrajectoryManager.t -> t
  val backward_iteration : t -> (Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t) -> 
                          (Tensor.t -> Tensor.t) -> Tensor.t array * Tensor.t array
end

module ValueFunctionSolver : sig
  type config = {
    state_dim: int;
    control_dim: int;
    time_step: float;
    terminal_time: float;
    space_steps: int array;
  }
  
  type grid_point = {
    state: Tensor.t;
    value: Tensor.t;
    gradient: Tensor.t option;
  }
  
  type t
  
  val create : config -> (Tensor.t -> Tensor.t -> Tensor.t) -> 
               (Tensor.t -> float -> Tensor.t) -> (Tensor.t -> Tensor.t) -> t
  val generate_grid : t -> {min: float; max: float} array -> grid_point array
  val solve_pde : t -> grid_point array -> grid_point array
end

module Configuration : sig
  type numerical_config = {
    time_step: float;
    terminal_time: float;
    space_steps: int array;
    basis_size: int;
    batch_size: int;
    solver_type: LeastSquaresSolvers.solver_type;
    integrator_type: NumericalIntegrator.integrator_type;
    tol: float;
  }

  type problem_config = {
    state_dim: int;
    control_dim: int;
    num_paths: int;
    use_importance_sampling: bool;
  }

  type t = {
    numerical: numerical_config;
    problem: problem_config;
  }

  val create : numerical_config -> problem_config -> t
  val standard_config : state_dim:int -> control_dim:int -> num_paths:int -> t
end

module FBSDESystem : sig
  type config = {
    state_dim: int;
    control_dim: int;
    time_step: float;
    terminal_time: float;
    num_paths: int;
    basis_size: int;
    batch_size: int;
  }

  type t
  
  val create : config -> (module Process) -> AdvancedLSMC.t -> 
               ValueFunctionSolver.t -> NumericalIntegrator.t -> 
               ImportanceSampling.t option -> PathMeasure.t -> t
               
  val solve : t -> Tensor.t -> (Tensor.t -> Tensor.t) -> 
              (Tensor.t -> float -> Tensor.t) -> 
              {
                value: Tensor.t;
                paths: Tensor.t array array;
                value_function: ValueFunctionSolver.grid_point array;
                y_evolution: Tensor.t array;
                z_evolution: Tensor.t array;
              }
end