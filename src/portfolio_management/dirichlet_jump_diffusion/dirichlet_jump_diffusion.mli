open Torch

type vector = Tensor.t
type matrix = Tensor.t

type kato_function = {
  value: vector -> float;
  dim: int;
  device: Device.t;
}

type jump_measure = {
  intensity: vector -> float;
  kernel: vector -> vector;
  is_finite: bool;
}

type boundary_info = {
  domain: vector -> bool;
  distance: vector -> float;
  is_regular: vector -> bool;
}

type solution = {
  value: vector -> float;
  gradient: vector -> vector option;
  is_weak: bool;
}

type operator_config = {
  diffusion: vector -> matrix;
  drift: vector -> vector;
  jump: jump_measure;
  dim: int;
  device: Device.t;
}

type path_state = {
  position: vector;
  time: float;
  alive: bool;
}

type path_info = {
  exit_time: float option;
  exit_position: vector option;
  hitting_position: vector option;
  did_jump: bool;
}

module Kato : sig
  val create : (vector -> float) -> int -> Device.t -> kato_function
  val verify_small_time : kato_function -> bool
  val verify_dimensional : kato_function -> bool
end

module CoreOperator : sig
  type t = {
    config: operator_config;
    device: Device.t;
  }
  
  val create : operator_config -> Device.t -> t
  val apply_l0 : t -> (vector -> float) -> vector -> Tensor.t
  val apply_jump : t -> (vector -> float) -> vector -> Tensor.t
  val apply : t -> (vector -> float) -> vector -> Tensor.t
end

module Measure : sig
  type t = {
    density: vector -> float;
    support: vector -> bool;
    dimension: int;
    device: Device.t;
  }

  val create : density:(vector -> float) -> 
               support:(vector -> bool) -> 
               dim:int -> 
               device:Device.t -> t

  module Integration : sig
    val monte_carlo : t -> (vector -> float) -> n_samples:int -> float
    val grid_integrate : t -> (vector -> float) -> 
                        bounds:{ high: float; low: float } -> 
                        grid_points:int -> float
  end

  module Operations : sig
    val product : t -> t -> t
    val restrict : t -> (vector -> bool) -> t
  end
end

module PathGenerator : sig
  type path = {
    states: path_state list;
    info: path_info;
  }

  val generate : CoreOperator.t -> x0:vector -> max_time:float -> dt:float -> path

  module Analysis : sig
    val analyze_path : path -> {
      n_jumps: int;
      max_jump_size: float;
      total_time: float;
    }
  end
end

module Boundary : sig
  val create : (vector -> bool) -> dim:int -> device:Device.t -> boundary_info
  val find_hitting_time : boundary_info -> PathGenerator.path -> (float * vector) option
end

module WeakSolution : sig
  type test_function = {
    value: vector -> float;
    gradient: vector -> vector;
    support: vector -> bool;
  }

  module TestFunctions : sig
    val create : vector -> float -> test_function
    val create_sequence : int -> int -> Device.t -> test_function list
  end

  val verify_solution : CoreOperator.t -> boundary_info -> solution -> test_function list -> bool
end

module DirichletSolver : sig
  type solver_config = {
    n_samples: int;
    max_time: float;
    time_step: float;
    tolerance: float;
  }

  val create : CoreOperator.t -> boundary_info -> solver_config -> 
               (CoreOperator.t * boundary_info * solver_config)
  val solve : (CoreOperator.t * boundary_info * solver_config) -> 
              (vector -> float) -> solution
  val verify : (CoreOperator.t * boundary_info * solver_config) -> solution -> bool
end

module ErrorAnalysis : sig
  type error_components = {
    spatial: float;
    temporal: float;
    statistical: float;
    total: float;
    convergence_rate: float option;
  }

  val analyze_error : solution -> solution -> CoreOperator.t -> boundary_info -> 
                     error_components
  val estimate_bounds : solution -> CoreOperator.t -> boundary_info -> float * float
end

module Convergence : sig
  type convergence_info = {
    spatial_rate: float;
    temporal_rate: float;
    statistical_rate: float;
    overall_rate: float;
    iterations: int;
    achieved_tolerance: float;
  }

  val analyze : solution -> CoreOperator.t -> boundary_info -> 
                DirichletSolver.solver_config -> convergence_info
end

module Utils : sig
  val generate_test_points : int -> int -> Device.t -> vector list
  val compute_statistics : float list -> float * float * float
  val estimate_convergence_rate : float list -> float option

  module BoundaryUtils : sig
    val is_near_boundary : vector -> boundary_info -> float -> bool
    val find_nearest_boundary : vector -> boundary_info -> vector option
  end
end