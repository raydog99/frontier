open Torch

type point = Tensor.t
type multi_index = int array

type coefficient = 
  | Constant of float
  | Function of (point -> float)
  | Tensor of Tensor.t

type linear_operator = {
  order: int;
  eval: Tensor.t -> point -> Tensor.t;
  adjoint: Tensor.t -> point -> Tensor.t;
  domain_dim: int;
  range_dim: int;
}

module type Kernel = sig
  type rkhs = {
    centers: point array;
    coefficients: Tensor.t array;
    length_scale: float;
  }

  val length_scale : float
  val evaluate : point -> point -> Tensor.t
  val gradient : point -> point -> Tensor.t
  val partial_derivative : multi_index -> point -> point -> Tensor.t
end

module FunctionSpace : sig
  type norm_type = L2 | H1 | H2 | Sobolev of int
  
  val compute_norm : norm_type -> (point -> Tensor.t) -> point array -> Tensor.t
  val test_embedding : (point -> Tensor.t) -> 
    source_space:norm_type -> 
    target_space:norm_type -> 
    test_points:point array -> 
    bool
end

val derivative_operator : (module Kernel) -> multi_index -> point -> point -> Tensor.t
val kernel_matrix : (module Kernel) -> point array -> multi_index array -> Tensor.t
val feature_matrix : (module Kernel) -> point array -> Tensor.t array -> multi_index array -> Tensor.t
val apply_regularization : Tensor.t -> float -> 
  [`Standard | `Tikhonov | `Adaptive of float] -> Tensor.t
val solve_regularized_system : (module Kernel) -> point array -> Tensor.t array -> 
  multi_index array -> float -> Tensor.t

(** Optimization methods *)
module OptimizationMethods : sig
  type optimization_method = 
    | GradientDescent of {step_size: float}
    | LBFGS of {m: int; max_line_search: int}
    | GaussNewton
    | TrustRegion of {radius: float; eta: float}

  val lbfgs_update : Tensor.t list -> Tensor.t list -> Tensor.t -> Tensor.t
  val gauss_newton_update : Tensor.t -> Tensor.t -> Tensor.t
  val trust_region_update : radius:float -> eta:float -> 
    Tensor.t -> Tensor.t -> Tensor.t
end

module RBFKernel : Kernel
module MaternKernel (P : sig val nu : float end) : Kernel
module PolynomialKernel (P : sig val degree : int end) : Kernel

module NonlinearPDE : sig
  type nonlinear_term =
    | Polynomial of float array
    | Rational of float array * float array
    | Composition of (Tensor.t -> Tensor.t) * linear_operator
    | Product of nonlinear_term list
    | Sum of nonlinear_term list

  val evaluate_term : nonlinear_term -> Tensor.t -> point -> Tensor.t
  val derivative_term : nonlinear_term -> Tensor.t -> Tensor.t -> point -> Tensor.t
end

module NonlinearPDESolver : sig
  type problem = {
    linear_operators: linear_operator array;
    nonlinear_terms: NonlinearPDE.nonlinear_term array;
    kernel: (module Kernel);
    collocation_points: point array;
    boundary_points: point array;
    forcing_term: point -> Tensor.t;
    boundary_data: point -> Tensor.t;
  }

  val compute_residual : problem -> Tensor.t -> point -> Tensor.t
  val compute_jacobian : problem -> Tensor.t -> Tensor.t
  val solve : problem -> init_guess:Tensor.t -> max_iter:int -> tol:float -> Tensor.t
end

module Regularization : sig
  type regularizer = 
    | L2 of float
    | H1 of float
    | H2 of float
    | TV of float
    | Composite of regularizer list

  val apply : regularizer -> Tensor.t -> Tensor.t
  val grad : regularizer -> Tensor.t -> Tensor.t
end

module UnknownCoefficient : sig
  type t = {
    kernel: (module Kernel);
    points: point array;
    values: Tensor.t array;
    derivatives: (multi_index * Tensor.t array) array;
  }

  val create : (module Kernel) -> point array -> Tensor.t array -> t
  val evaluate : t -> point -> Tensor.t
  val evaluate_derivative : t -> point -> multi_index -> Tensor.t
end

module Framework : sig
  type config = {
    kernel_type: [ 
      | `RBF of float 
      | `Matern of float * float
      | `Polynomial of int * float * float
    ];
    regularization: [`Standard | `Tikhonov | `Adaptive of float];
    optimization_method: OptimizationMethods.optimization_method;
    max_iter: int;
    tol: float;
  }

  val create_kernel : config -> (module Kernel)
  val solve : config:config -> 
    operators:linear_operator array ->
    nonlinear_terms:NonlinearPDE.nonlinear_term array ->
    points:point array ->
    boundary_points:point array ->
    forcing:(point -> Tensor.t) ->
    boundary:(point -> Tensor.t) ->
    Tensor.t
end