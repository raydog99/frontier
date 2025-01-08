open Torch

type time = float
type value = float array
type activity = float array

type process_state = {
  time: time;
  values: value;
  activity: activity;
}

module type Process = sig
  type t
  val create : activity -> t
  val step : t -> float -> t
  val get_state : t -> process_state
end

(* Generalized Liouville Process *)
module GLP : sig
  val from_levy_bridge : float array -> float -> t
  val get_terminal_distribution : t -> float array
  val compute_increments : t -> float -> float array
  val update : t -> float -> float array -> t
end

(* Lévy Bridge *)
module LevyBridge : sig
  type t
  val create : (float -> float) -> float -> t
  val transition_density : t -> float -> float -> float -> float -> float
  val get_state : t -> process_state
end

(* Brownian Liouville Process *)
module BLP : sig
  val create_with_sigma : activity -> float -> t
  val get_bridges : t -> float array
  val compute_covariance : t -> float array array
end

(* Poisson Liouville Process *)
module PLP : sig
  val create_with_intensity : activity -> float -> t
  val compute_intensity : t -> float
  val generate_jumps : t -> float -> int array
end

module MeasureChange : sig
  type density_process
  val create_density_process : process_state -> density_process
  val change_measure : process_state -> process_state
  val compute_radon_nikodym : density_process -> float
end

module MartingaleChar : sig
  type characterization = {
    is_martingale: bool;
    local_martingale: bool;
    predictable: bool;
    angle_bracket: float array array;
  }
  val characterize_process : process_state -> (float -> float) -> characterization
end

module PathProperties : sig
  type path = {
    times: float array;
    values: float array array;
  }
  val compute_holder_exponent : path -> float array
  val detect_jumps : path -> float -> (float * float array) list
  val compute_occupation_density : path -> int -> float array array
end

module Numerical : sig
  type scheme = Euler | Milstein | RK4
  val solve : process_state -> scheme -> float -> int -> 
    (float * float array) list
  val adaptive_step : process_state -> float -> float * float array
end

module Simulation : sig
  type config = {
    n_paths: int;
    n_steps: int;
    dt: float;
    scheme: Numerical.scheme;
    antithetic: bool;
    stratification: int option;
  }

  val simulate_paths : GLP.t -> config -> PathProperties.path array
  
  module QuasiMonteCarlo : sig
    val sobol : int -> int -> float array array
  end
  
  module MultiLevel : sig
    type level = {
      dt: float;
      samples: float array array;
      correction: float array array;
    }
    val simulate : GLP.t -> config -> int -> float array
  end
end

module RiskMeasures : sig
  type risk_measure = {
    var: float array;
    cvar: float array;
    expected_shortfall: float array;
    maximum_drawdown: float array;
  }
  
  val compute_var : float array array -> float -> float array
  val compute_cvar : float array array -> float -> float array
  val compute_risk_measures : float array array -> float -> risk_measure
end

module Analysis : sig
  type dependency_measure = {
    correlation: float array array;
    rank_correlation: float array array;
    tail_dependence: float array array;
    copula_estimate: float array array array;
  }

  val compute_correlation : float array array -> float array array
  val estimate_copula : float array array -> int -> float array array array
  val analyze_dependencies : float array array -> dependency_measure
end

module Calibration : sig
  type calibration_result = {
    parameters: float array;
    error: float;
    iterations: int;
    convergence: bool;
  }

  type optimization_method = [
    | `GradientDescent of {
        learning_rate: float;
        momentum: float;
        max_iter: int;
        tolerance: float;
      }
    | `SimulatedAnnealing of {
        temp_schedule: int -> float;
        max_iter: int;
      }
  ]

  val calibrate : GLP.t -> float array array -> optimization_method -> calibration_result
end

module StochasticIntegral : sig
  type decomposition = {
    drift: float array;
    martingale: float array;
    compensator: float array;
  }

  val compute_decomposition : process_state -> decomposition
  val verify_harness : process_state -> (float * float * float * float) -> bool
end

val check_weak : process_state -> float array -> int -> bool
val check_strong : process_state -> float array -> int -> bool