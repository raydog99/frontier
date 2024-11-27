open Torch

module NumericalSchemes : sig
  (** Spatial discretization types *)
  type spatial_scheme =
    | CentralDiff    (** Second-order central differences *)
    | UpwindDiff     (** First-order upwind *)
    | WENO of int    (** WENO scheme of specified order *)
    | ENO of int     (** ENO scheme of specified order *)

  (** Time integration types *)
  type time_scheme =
    | ExplicitEuler
    | ImplicitEuler
    | CrankNicolson
    | RK4           (** Classical 4th order Runge-Kutta *)
    | IMEX of int   (** IMEX scheme of specified order *)

  (** Discretization parameters *)
  type discretization_params = {
    spatial_scheme: spatial_scheme;
    time_scheme: time_scheme;
    dx: float array;     (** Spatial steps for each dimension *)
    dt: float;          (** Time step *)
    theta: float;       (** Parameter for theta-methods *)
  }

  (** WENO reconstruction *)
  val weno_reconstruction : int -> float array -> float

  (** ENO reconstruction *)
  val eno_reconstruction : int -> float array -> float

  (** IMEX time stepping *)
  val imex_step : discretization_params -> (Tensor.t -> Tensor.t) -> (Tensor.t -> Tensor.t) -> Tensor.t -> float -> Tensor.t

  (** RK4 time stepping *)
  val rk4_step : discretization_params -> (Tensor.t -> Tensor.t) -> Tensor.t -> float -> Tensor.t

  (** Full solver implementation *)
  val solve : discretization_params -> Tensor.t -> float -> Tensor.t
end

(** Module for sophisticated boundary condition handling *)
module BoundaryHandler : sig
  (** Boundary condition types *)
  type boundary_type =
    | Dirichlet of float  (** Fixed value *)
    | Neumann of float    (** Fixed derivative *)
    | Robin of {          (** Robin boundary condition *)
        alpha: float;
        beta: float;
        gamma: float;
      }
    | Periodic            (** Periodic boundary *)
    | Artificial of {     (** Artificial boundary *)
        order: int;
        extrapolation: bool;
      }

  (** Apply boundary conditions to solution *)
  val apply_boundary : Tensor.t -> discretization_params -> boundary_type array -> float array -> int -> Tensor.t
end

(** Module for mixed derivative approximations *)
module MixedDerivatives : sig
  (** Mixed derivative approximation types *)
  type mixed_approx =
    | CentralDiff       (** Standard central difference *)
    | FourthOrder      (** Fourth-order accurate *)
    | CompactStencil   (** Compact 9-point stencil *)
    | Upwind           (** Direction-biased stencil *)

  (** Compute mixed derivative approximations *)
  val compute_mixed_derivative : mixed_approx -> Tensor.t -> float array -> int -> int -> Tensor.t

  (** Build mixed derivative operator matrix *)
  val build_operator : discretization_params -> float array -> int -> int -> Tensor.t
end

(** Module for stability analysis *)
module StabilityAnalysis : sig
  (** Stability analysis types *)
  type stability_method =
    | VonNeumann         (** von Neumann stability analysis *)
    | MatrixStability    (** Matrix-based stability analysis *)
    | EnergyMethod       (** Energy method stability *)
    | CFLCondition       (** CFL condition checking *)

  (** Stability analysis parameters *)
  type stability_params = {
    method_type: stability_method;
    max_time: float;
    space_steps: int array;
    time_steps: int;
    safety_factor: float;
  }

  (** Compute stability condition *)
  val analyze_stability : stability_params -> Tensor.t -> bool

  (** Suggest stable parameters *)
  val suggest_parameters : stability_params -> Tensor.t -> float * float array
end

(** Module for forward rate properties *)
module ForwardRateProperties : sig
  (** Check forward rate martingale property under appropriate measure *)
  val check_martingale : discretization_params -> int -> Measures.measure -> float -> bool

  (** Forward measure change *)
  val change_measure_dynamics : discretization_params -> float -> float -> Measures.measure -> Measures.measure -> Tensor.t
end

(** Module for correlation structure handling *)
module CorrelationStructure : sig
  type correlation_config = {
    base_correlation: float;
    time_decay: float option;
    tenor_decay: float option;
    custom_correlation: (int -> int -> float) option;
  }

  (** Build correlation matrix *)
  val build_correlation : correlation_config -> int -> Tensor.t

  (** Generate correlated increments *)
  val generate_increments : Tensor.t -> int -> Tensor.t
end

(** Module for complete PDE system handling *)
module CompletePDESystem : sig
  (** Complete drift computation *)
  val compute_drift : discretization_params -> float array -> float -> Tensor.t

  (** Complete PDE system matrix assembly *)
  val build_system_matrix : discretization_params -> float array -> float -> Tensor.t

  (** Complete relative price PDE *)
  val relative_price_pde : discretization_params -> float array -> float -> Tensor.t * Tensor.t
end

(** Module for complete PDE solver *)
module CompletePDE : sig
  (** Full parabolic PDE operator *)
  type pde_operator = {
    time_derivative: Tensor.t -> Tensor.t;
    space_derivatives: Tensor.t -> Tensor.t;
    mixed_derivatives: Tensor.t -> Tensor.t;
    boundary_conditions: Tensor.t -> Tensor.t;
  }

  (** Build complete PDE operator *)
  val build_operator : discretization_params -> float array -> float -> pde_operator

  (** Solve complete PDE system *)
  val solve : discretization_params -> (float array -> float) -> Tensor.t
end

(** Module for complete discretization schemes *)
module PDEDiscretization : sig
  (** Finite difference coefficients *)
  type fd_coefficients = {
    dx: float;
    dt: float;
    theta: float;  (** For theta-methods *)
  }

  (** First derivative - central difference *)
  val first_derivative : Tensor.t -> int -> Tensor.t

  (** Second derivative *)
  val second_derivative : Tensor.t -> int -> Tensor.t

  (** Mixed derivative *)
  val mixed_derivative : Tensor.t -> int -> int -> Tensor.t
end

(** Module for complete swaption valuation *)
module CompleteSwaptionMethods : sig
  (** IRS valuation *)
  val compute_irs_value : discretization_params -> float array -> float -> float -> float -> float

  (** Tenor structure handling *)
  type tenor_grid = {
    payment_dates: float array;
    fixing_dates: float array;
    observation_dates: float array;
    year_fractions: float array;
  }

  (** Build tenor grid *)
  val build_tenor_grid : discretization_params -> float -> float -> tenor_grid

  (** Price swaption *)
  val price_swaption : discretization_params -> float array -> tenor_grid -> float -> Tensor.t

  (** Price forward-starting swaption *)
  val price_forward_starting_swaption : discretization_params -> float array -> tenor_grid -> float -> float -> Tensor.t

  (** Compute swaption sensitivities *)
  val compute_greeks : discretization_params -> float array -> tenor_grid -> float -> {
    delta: float array;
    gamma: float array;
    vega: float array;
  }
end

(** Module for complete Monte Carlo methods *)
module CompleteMonteCarloMethods : sig
  (** Monte Carlo simulation parameters *)
  type mc_params = {
    n_paths: int;
    time_steps: int;
    dt: float;
    variance_reduction: bool;
    antithetic: bool;
    seed: int option;
  }

  (** Evolve rates with volatility decay *)
  val evolve_rates : discretization_params -> float array -> float -> float -> float array

  (** Generate complete path with measure changes *)
  val generate_path : discretization_params -> float array -> mc_params -> Tensor.t

  (** Simulate multiple paths *)
  val simulate : discretization_params -> mc_params -> Tensor.t

  (** Price swaption with Monte Carlo *)
  val price_swaption : discretization_params -> tenor_grid -> float -> mc_params -> {
    price: float;
    confidence_interval: float * float;
    std_error: float;
  }

  (** Error analysis module *)
  module ErrorAnalysis : sig
    val analyze_convergence : discretization_params -> tenor_grid -> float -> {
      results: {
        price: float;
        confidence_interval: float * float;
        std_error: float;
      } array;
      rates: float array;
      limit: float;
    }
  end
end

(** Module for multi-dimensional PDE solver *)
module MultiDimPDESolver : sig
  (** Spatial discretization parameters *)
  type spatial_params = {
    points: int array;        (** Grid points per dimension *)
    min_rates: float array;   (** Minimum rate values *)
    max_rates: float array;   (** Maximum rate values *)
    transform_type: [
      | `Uniform             (** Uniform grid *)
      | `Sinh of float       (** Sinh transformation with concentration param *)
      | `Adaptive           (** Adaptive grid points *)
    ]
  }

  (** Build multi-dimensional grid *)
  val build_grid : discretization_params -> spatial_params -> float array array

  (** ADI scheme implementation *)
  module ADI : sig
    val douglas_step : PDEFormulation.coefficients -> float -> Tensor.t -> float -> Tensor.t
  end

  (** Complete multi-dimensional solver *)
  val solve : discretization_params -> spatial_params -> (float array -> float) -> Tensor.t
end

(** Module for adaptive mesh handling *)
module AdaptiveMesh : sig
  (** Compute monitor function for mesh adaptation *)
  val compute_monitor_function : discretization_params -> float array -> int -> float array

  (** Redistribute points based on monitor function *)
  val redistribute_points : float array -> float array -> float array

  (** Estimate error for adaptation *)
  val estimate_error : float array -> float array -> float array

  (** Full mesh adaptation cycle *)
  val adapt_mesh : discretization_params -> float array -> float array -> float -> (float array * float array) option
end

(** Module for discount factor computations *)
module DiscountFactors : sig
  (** Compute P(Ti,Tj) *)
  val compute_discount_factor : discretization_params -> float array -> float -> float -> Tensor.t

  (** Compute bank account *)
  val compute_bank_account : discretization_params -> float array -> float -> Tensor.t
end

(** Module for generalized forward measure handling *)
module GeneralizedFMM : sig
  (** Compute volatility decay *)
  val compute_volatility_decay : float -> float -> float -> float

  (** Compute drift under different measures *)
  val compute_drift : discretization_params -> float array -> float -> Measures.measure -> Tensor.t

  (** Evolve rates with volatility decay *)
  val evolve_rates : discretization_params -> float array -> float -> float -> Measures.measure -> float array
end