open Torch

type state = Tensor.t
type control = Tensor.t
type time = float
type measure = Tensor.t

val beta : float
val r : float
val k : float

module Measure : sig
  type t = {
    density: Tensor.t;
    support: float * float;
    grid_points: int;
  }

  val create : Tensor.t -> (float * float) -> int -> t

  module Topology : sig
    val check_weak_convergence : t -> t -> (int -> Tensor.t) array -> float
    val total_variation : t -> t -> float
    val wasserstein_distance : t -> t -> float
  end

  module Derivatives : sig
    val first_variation : t -> (t -> float) -> Tensor.t
    val second_variation : t -> (t -> float) -> Tensor.t
    val lions_derivative : t -> (t -> float) -> Tensor.t * Tensor.t
  end

  module Transform : sig
    val pushforward : t -> (Tensor.t -> Tensor.t) -> t
    val check_regularity : t -> bool
  end
end

module Stochastic : sig
  type filtration = {
    time_grid: float array;
    events: Tensor.t array;
    is_complete: bool;
    is_right_continuous: bool;
  }

  type 'a process = {
    paths: 'a array;
    filtration: filtration;
    time_index: int array;
  }

  val create_complete_filtration : float array -> filtration

  module Integration : sig
    val quadratic_variation : 'a process -> Tensor.t array
    val ito_integral : Tensor.t array -> 'a process -> float -> Tensor.t array
    val stratonovich_integral : Tensor.t array -> 'a process -> float -> Tensor.t array
  end

  module Martingale : sig
    val is_martingale : Tensor.t process -> bool
  end

  module Adaptedness : sig
    val is_adapted : Tensor.t process -> bool
    val is_progressively_measurable : Tensor.t process -> bool
    val adapt : Tensor.t process -> Tensor.t process
  end

  module Predictable : sig
    val is_predictable : Tensor.t process -> bool
  end
end

module Control : sig
  type infinite_sequence = {
    coefficients: Tensor.t;
    r_values: Tensor.t;
    truncation_index: int;
  }

  type control = {
    sequence: infinite_sequence;
    bound: float;
    support: float * float;
  }

  val create_sequence : Tensor.t -> Tensor.t -> int -> infinite_sequence option
  val create_control : infinite_sequence -> float -> (float * float) -> control option

  module Evaluation : sig
    val evaluate_h : control -> Tensor.t -> Tensor.t
    val evaluate_derivative : control -> Tensor.t -> Tensor.t
  end

  module Topology : sig
    val product_distance : control -> control -> Tensor.t
    val check_convergence : control array -> bool * float
  end
end

module MVM : sig
  type mvm_state = {
    measure: Measure.t;
    filtration: Stochastic.filtration;
    time: float;
    path: Measure.t array;
  }

  val create : Measure.t -> Stochastic.filtration -> mvm_state

  module Evolution : sig
    val evolve : mvm_state -> Control.control -> Tensor.t -> float -> mvm_state
    val verify_martingale : mvm_state -> (Tensor.t -> Tensor.t) -> bool
  end

  module Properties : sig
    val verify_properties : mvm_state -> bool * bool
  end
end

module HJB : sig
  type value_function = {
    value: Measure.t -> float;
    gradient: Measure.t -> Tensor.t;
    hessian: Measure.t -> Tensor.t Tensor.t;
  }

  type hjb_solution = {
    value_function: value_function;
    optimal_control: Measure.t -> Control.control option;
    verification: Measure.t -> bool;
  }

  module Hamiltonian : sig
    val compute_sigma : Measure.t -> Control.control -> Tensor.t
    val compute : Measure.t -> float -> Tensor.t -> Tensor.t Tensor.t -> Control.control -> float
    val optimize : Measure.t -> float -> Tensor.t -> Tensor.t Tensor.t -> float * Control.control option
  end

  module Viscosity : sig
    type test_function = {
      phi: Measure.t -> float;
      grad_phi: Measure.t -> Tensor.t;
      hess_phi: Measure.t -> Tensor.t Tensor.t;
    }

    val verify_subsolution : value_function -> test_function -> Measure.t -> bool
    val verify_supersolution : value_function -> test_function -> Measure.t -> bool
    val create_test_functions : Measure.t -> test_function array
  end

  val solve : Measure.t -> float -> hjb_solution
end

module DP : sig
  type value_function = {
    value: Measure.t -> float;
    gradient: Measure.t -> Tensor.t;
    hessian: Measure.t -> Tensor.t Tensor.t;
  }

  module Operator : sig
    val apply_t_operator : value_function -> Measure.t -> Control.control -> float -> float
    val apply_optimal_t : value_function -> Measure.t -> float -> float * Control.control option
  end

  module ValueIteration : sig
    val iteration_step : value_function -> Measure.t -> float -> value_function * Control.control option
    val iterate : value_function -> Measure.t -> int -> float -> float -> value_function
  end

  module PolicyIteration : sig
    val evaluate_policy : Control.control -> value_function -> Measure.t -> float -> float
    val improve_policy : value_function -> Measure.t -> float -> Control.control option
    val iterate : Control.control -> value_function -> Measure.t -> int -> float -> float -> Control.control * value_function
  end
end

module ConvergenceAnalysis : sig
  type divergence_metrics = {
    value_divergence: float;
    gradient_divergence: float;
    control_divergence: float;
    total_variation: float;
    wasserstein: float;
  }

  module ValueConvergence : sig
    val compute_value_divergence : DP.value_function -> DP.value_function -> Measure.t -> divergence_metrics
    val check_cauchy_sequence : DP.value_function array -> Measure.t -> bool * divergence_metrics array
  end

  module DivergencePropagation : sig
    val analyze_value_iteration : DP.value_function -> Measure.t -> Control.control array -> float -> int -> divergence_metrics array
    val compute_stability_bounds : divergence_metrics array -> float * float * float
  end

  module MeasureConvergence : sig
    val compute_rate : Measure.t -> Measure.t -> float -> float
    val analyze_sequence : Measure.t array -> float -> float * float array
  end
end

module Cost : sig
  type cost_function = {
    running_cost: Measure.t -> Control.control -> float;
    terminal_cost: Measure.t -> float option;
  }

  val create_bounded_cost : float -> cost_function
  val evaluate : Measure.t -> Control.control -> Tensor.t
  val compute_discounted_cost : cost_function -> MVM.mvm_state -> Control.control -> float -> float
end