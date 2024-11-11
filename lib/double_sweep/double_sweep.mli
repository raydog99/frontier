open Torch

module Types : sig
  type grid_params = {
    n_time: int;
    n_space: int;
    x_min: float;
    x_max: float;
    t_max: float;
    grid_type: grid_type;
  }
  
  and grid_type = 
    | Uniform
    | Hyperbolic of float
    | Square_root

  type market_params = {
    r: float;
    mu: float;
    sigma: float;
    s0: float;
    k: float;
    is_time_dependent: bool;
  }

  type solve_status = {
    converged: bool;
    iterations: int;
    error: float;
  }
end

module BlackScholes : sig
  val make_grid : Types.grid_params -> grid_type -> Tensor.t * Tensor.t * float * float
  val build_coefficients : Types.grid_params -> Types.market_params -> Tensor.t -> float -> 
    Tensor.t * Tensor.t * Tensor.t
  val initial_conditions : Types.grid_params -> Types.market_params -> Tensor.t -> bool -> Tensor.t
  val apply_boundary_conditions : Tensor.t -> Tensor.t -> float -> Types.market_params -> bool -> Tensor.t
end

module ErrorAnalysis : sig
  type error_component = {
    truncation: float;
    roundoff: float;
    iteration: float;
    boundary: float;
  }

  type stability_metric = {
    spectral_radius: float;
    condition_number: float;
    growth_factor: float;
    dissipation_error: float;
    dispersion_error: float;
  }

  val analyze_truncation_error : Tensor.t -> float -> float -> Tensor.t
  val analyze_stability : Tensor.t -> Tensor.t -> Tensor.t -> float -> stability_metric
  val estimate_total_error : Tensor.t -> float -> float -> stability_metric -> Tensor.t
end


module TRBDF2 : sig
  val alpha : float
  
  val trapezoidal_stage : Tensor.t -> (Tensor.t -> Tensor.t) -> float -> Tensor.t
  val bdf2_stage : Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t) -> float -> Tensor.t
  val step : float -> Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t -> Types.market_params -> 
    Tensor.t * float
end

module IlinScheme : sig
  type scheme_params = {
    beta: float;
    use_limiting: bool;
    stabilization: [`None | `SUPG | `Full];
  }

  val compute_coefficients : float array -> (float -> float) -> (float -> float) -> float -> float -> scheme_params ->
    float array * float array * float array
  val solve : float array -> (float -> float) -> (float -> float) -> float -> float -> scheme_params -> 
    float array -> float array -> Tensor.t
end

module DoubleSweepLU : sig
  type sweep_result = {
    solution: Tensor.t;
    barrier_indices: (int * int) option;
    convergence: Types.solve_status;
  }

  val solve : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> 
    sweep_result
  val solve_two_barriers : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> 
    sweep_result
end


module MMatrixVerification : sig
  type violation_type =
    | DiagonalNonPositive
    | OffDiagonalNonNegative
    | NonDominantDiagonal
    | Reducible
    | BarrierConditionViolation

  type violation = {
    kind: violation_type;
    location: int;
    magnitude: float;
    details: string;
  }

  val verify_m_matrix_properties : Tensor.t -> Tensor.t -> Tensor.t -> violation list
  val verify_barrier_conditions : Tensor.t -> Tensor.t -> Tensor.t -> float -> violation list
  val verify_all : Tensor.t -> Tensor.t -> Tensor.t -> float -> violation list
end

module TRBDFEigenAnalysis : sig
  type eigenvalue_result = {
    max_real: float;
    max_imag: float;
    condition_number: float;
    stability_region: (float * float) list;
  }

  val solve_trbdf2_stability : Tensor.t -> float -> eigenvalue_result
  val power_method : Tensor.t -> int -> float -> float * int * bool
  val compute_eigenspectrum : Tensor.t -> int -> float -> float list
end


module IntersectionPointHandler : sig
  type intersection_detail = {
    time: float;
    location: int * int;
    precise_point: float * float;
    pre_velocities: float * float;
    post_behavior: [`Continuous | `Discontinuous of float];
  }

  val find_intersection_point : Tensor.t -> Tensor.t -> float -> float -> Tensor.t -> 
    intersection_detail option
  val handle_intersection_region : Tensor.t -> Tensor.t -> float -> intersection_detail option -> 
    Tensor.t
  val apply_intersection_barrier_conditions : Tensor.t -> Tensor.t -> float -> 
    intersection_detail option -> (float * float) option
end

module IntersectionTimeStepping : sig
  val adapt_timestep : float -> IntersectionPointHandler.intersection_detail option -> float
  val solve_near_intersection : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t -> 
    Tensor.t -> float -> float -> IntersectionPointHandler.intersection_detail option -> 
    (Tensor.t, string) result
end

module IntersectionErrorAnalysis : sig
  type error_metrics = {
    max_error: float;
    l2_error: float;
    intersection_error: float option;
    smoothness_violation: float option;
    barrier_preservation: float;
  }

  val compute_intersection_errors : Tensor.t -> Tensor.t -> float -> 
    IntersectionPointHandler.intersection_detail option -> Tensor.t -> error_metrics
end


module BarrierSolver : sig
  type barrier_point = {
    location: int;
    value: float;
    time: float;
    derivative: float;
    second_derivative: float;
  }

  type intersection_state = {
    time: float;
    location: float * float;
    pre_intersection_slopes: float * float;
    post_intersection_behavior: [`Continuous | `Discontinuous of float];
    stability_metric: float;
  }

  val compute_barrier_derivatives : Tensor.t -> Tensor.t -> float -> float * float
  val locate_barrier : Tensor.t -> Tensor.t -> Tensor.t -> float -> barrier_point list
  val solve_intersection : barrier_point list -> barrier_point list -> float -> intersection_state option
  val solve_barrier_evolution : barrier_point list -> float -> (float * float * float) list
end