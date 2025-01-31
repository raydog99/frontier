open Torch

(** Core model types *)
type matrix = Tensor.t
type vector = Tensor.t

type model_params = {
  dimension: int;         (** d assets *)
  horizon: float;         (** T *)
  r_matrix: matrix;       (** R: mean reversion *)
  v_matrix: matrix;       (** V: volatility *)
  k_matrix: matrix;       (** K: permanent impact *)
  s_bar: vector;         (** S̄: long-term mean *)
  eta: matrix;           (** η: temporary impact *)
  gamma: float;          (** γ: risk aversion *)
  terminal_penalty: matrix;  (** Γ: terminal penalty *)
}

type state = {
  inventory: vector;  (** q: inventory *)
  price: vector;     (** S: price *)
  cash: float;       (** X: cash *)
  time: float;       (** t: time *)
}

type riccati_matrices = {
  a_matrix: matrix;  (** A *)
  b_matrix: matrix;  (** B *)
  c_matrix: matrix;  (** C *)
  d_vector: vector;  (** D *)
  e_vector: vector;  (** E *)
  f_scalar: float;   (** F *)
}

type solution = {
  riccati: riccati_matrices list;
  time_grid: float list;
  optimal_controls: Tensor.t list;
  state_trajectory: state list;
  value_trajectory: float list;
}

val is_positive_definite : matrix -> bool
val solve_lyapunov : matrix -> matrix -> matrix

(** Measures *)
module Measures : sig
  type measure = private {
    sample_space: vector -> bool;
    sigma_algebra: (vector -> bool) list;
    measure_fn: vector -> float;
  }

  type filtration_t = private {
    time_index: float;
    events: (vector -> bool) list;
    parent: measure option;
  }

  val create_probability_space : int -> measure
  val generate_filtration : measure -> float list -> filtration_t list
end

(** Brownian motion *)
module BrownianMotion : sig
  type brownian_path = {
    times: float array;
    values: Tensor.t;
    dimension: int;
  }

  val generate_path : int -> int -> float -> brownian_path
  val sample_at_time : brownian_path -> float -> vector option
end

(** State evolution *)
module StateEvolution : sig
  val temporary_impact : model_params -> vector -> float
  val permanent_impact : model_params -> vector -> vector
  val update_state : state -> vector -> float -> model_params -> state
  val terminal_value : state -> model_params -> float
end

(** Numerical methods *)
module AdaptiveMethods : sig
  type refinement_criteria = {
    error_threshold: float;
    max_level: int;
    coarsen_threshold: float;
    refine_threshold: float;
  }

  val estimate_local_error : Tensor.t -> float -> float -> float
  val adapt_mesh : float array array -> float array array -> refinement_criteria -> 
                  float array array * int array array
end

(** Riccati solver *)
module RiccatiSolver : sig
  type riccati_system = private {
    q_matrix: Tensor.t;
    y_matrix: Tensor.t;
    u_matrix: Tensor.t;
    terminal: Tensor.t;
    dimension: int;
  }

  val create_system : model_params -> riccati_system
  val solve_backward : riccati_system -> model_params -> float -> int -> 
                      (float * Tensor.t) list
end

(** HJB solver *)
module HJBSolver : sig
  type verification_result = {
    viscosity_property: bool;
    comparison_principle: bool;
    boundary_satisfied: bool;
    terminal_satisfied: bool;
  }

  type hjb_solution = {
    value_fn: state -> float;
    optimal_control: state -> vector;
    verification: verification_result;
  }

  val create_hjb_solution : model_params -> (float * Tensor.t) list -> hjb_solution
end

(** Optimal control *)
module OptimalControl : sig
  type feedback_control = {
    feedback_fn: state -> float -> vector;
    state_constraints: state -> bool;
    stability_verified: bool;
  }

  type control_strategy = {
    compute_control: state -> vector;
    verify_admissible: vector -> bool;
    feedback_law: feedback_control;
  }

  val create_strategy : model_params -> (float * Tensor.t) list -> control_strategy
end

(** Simulation *)
module Simulation : sig
  type simulation_params = {
    dt: float;
    n_steps: int;
    n_paths: int;
    seed: int option;
  }

  type execution_metrics = {
    total_costs: float list;
    avg_price_impact: float list;
    inventory_profiles: vector list;
    execution_shortfall: float list;
  }

  type simulation_result = {
    paths: state list list;
    controls: vector list list;
    values: float list list;
    metrics: execution_metrics;
  }

  val simulate_ou_price : model_params -> simulation_params -> vector array array
  val simulate_execution : OptimalControl.control_strategy -> model_params -> 
                         simulation_params -> simulation_result
end

(** Complete execution system *)
module ExecutionSystem : sig
  type execution_params = {
    model: model_params;
    simulation: Simulation.simulation_params;
    control: OptimalControl.control_strategy;
    verification: HJBSolver.verification_result;
  }

  type performance_metrics = {
    total_cost: float;
    implementation_shortfall: float;
    tracking_error: float;
    information_ratio: float;
  }

  type execution_result = {
    solution: solution;
    simulation: Simulation.simulation_result;
    performance: performance_metrics;
  }

  val create_execution_system : model_params -> execution_params
  val execute : execution_params -> state -> execution_result
end