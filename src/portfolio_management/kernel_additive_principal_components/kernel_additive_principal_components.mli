open Torch

(* Core types *)
type kernel = Tensor.t -> Tensor.t -> Tensor.t
type transformation = Tensor.t -> Tensor.t

(* Numerical Stability module *)
module NumericStability : sig
  val epsilon : float
  val max_condition_number : float
  val stable_inverse : Tensor.t -> Tensor.t
  val stable_symeig : ?eigenvectors:bool -> Tensor.t -> Tensor.t * Tensor.t
  val stable_sqrtm : Tensor.t -> Tensor.t
  val condition_number : Tensor.t -> Tensor.t
end

(* Kernel operations *)
module Kernel : sig
  val rbf_kernel : float -> kernel
  val linear_kernel : kernel
  val poly_kernel : int -> float -> kernel
  val create_kernel_matrix : kernel -> Tensor.t -> Tensor.t
  val center_kernel_matrix : Tensor.t -> Tensor.t
  val kernel_alignment : kernel -> kernel -> Tensor.t -> Tensor.t
end

(* Population statistics *)
module Population : sig
  val covariance : Tensor.t -> Tensor.t -> float
  val inner_product : transformation -> transformation -> Tensor.t -> float
  val variance : Tensor.t -> float
  val center : Tensor.t -> Tensor.t
  val normalize : Tensor.t -> Tensor.t
end

(* Measure theory *)
module Measure : sig
  type measurable_set = 
    | Empty
    | Full
    | Interval of float * float
    | Union of measurable_set list
    | Intersection of measurable_set list
    | Complement of measurable_set

  type 'a measure = private {
    space: 'a;
    measure: measurable_set -> float;
    support: measurable_set;
  }

  type measure_type =
    | Lebesgue
    | Counting
    | Gaussian of float * float
    | EmpiricalMeasure of Tensor.t
    | ProductMeasure of measure_type list

  val create_measure : measure_type -> 'a measure
end

(* Function spaces *)
module FunctionSpace : sig
  type function_space_type =
    | L2Space of Measure.measure_type
    | SobolevSpace of int * Measure.measure_type
    | InfiniteDimensional of (int -> float)

  type 'a function_space = private {
    space_type: function_space_type;
    inner_product: ('a -> 'a -> float);
    norm: ('a -> float);
    basis: (int -> 'a option);
  }

  val create_l2_space : Measure.measure_type -> 'a function_space
  val create_sobolev_space : int -> Measure.measure_type -> 'a function_space

  module InfiniteBasis : sig
    type basis_type =
      | Fourier
      | Hermite
      | Legendre
      | Custom of (int -> float -> float)

    val get_basis_function : basis_type -> (int -> float -> float)
    val create_truncated_basis : basis_type -> int -> (int -> float -> float) array
  end
end

(* Hilbert spaces *)
module HilbertSpace : sig
  type hilbert_space = private {
    inner_product: Tensor.t -> Tensor.t -> float;
    norm: Tensor.t -> float;
    dim: int option;
  }

  val create_finite_dimensional : int -> hilbert_space
  val create_infinite_dimensional : (Tensor.t -> Tensor.t -> float) -> hilbert_space
  val project_onto_subspace : hilbert_space -> Tensor.t array -> Tensor.t -> Tensor.t
end

(* Optimization *)
module Optimization : sig
  type constraint_type =
    | EqualityConstraint of (Tensor.t -> Tensor.t)
    | InequalityConstraint of (Tensor.t -> Tensor.t)
    | NormConstraint of float

  type optimization_params = {
    max_iter: int;
    tolerance: float;
    learning_rate: float;
    momentum: float;
    constraint_weight: float;
  }

  val default_params : optimization_params

  val project_onto_constraints : constraint_type list -> Tensor.t -> Tensor.t

  val trust_region :
    objective:(Tensor.t -> Tensor.t) ->
    gradient:(Tensor.t -> Tensor.t) ->
    hessian:(Tensor.t -> Tensor.t) ->
    ?params:optimization_params ->
    Tensor.t ->
    Tensor.t

  val augmented_lagrangian :
    objective:(Tensor.t -> Tensor.t) ->
    constraints:constraint_type list ->
    ?params:optimization_params ->
    Tensor.t ->
    Tensor.t
end

(* Numerical integration *)
module Integration : sig
  type integration_method =
    | Trapezoidal
    | Simpson
    | GaussLegendre of int
    | AdaptiveQuadrature
    | MonteCarloIntegration of int

  type error_estimate = {
    absolute_error: float;
    relative_error: float;
    n_evaluations: int;
  }

  val integrate :
    ?method_:integration_method ->
    ?tol:float ->
    (float -> float) ->
    float ->
    float ->
    error_estimate * float

  val integrate_multiple :
    (float array -> float) ->
    (float * float) array ->
    float
end

(* Special cases for KAPC *)
module SpecialCases : sig
  module CCA : sig
    type cca_result = {
      correlation: float;
      transform1: transformation;
      transform2: transformation;
    }

    val from_kapc : kernel array -> float array -> Tensor.t -> cca_result
  end

  module MultipleKernels : sig
    type kernel_weights = private {
      kernels: kernel array;
      weights: Tensor.t;
    }

    val optimize_weights : kernel array -> Tensor.t -> kernel_weights
    val create_composite : kernel_weights -> kernel
  end
end

(* Kernel additive principal components *)
module KAPC : sig
  type t

  val create : kernel array -> float array -> Tensor.t -> t
  val compute_objective : t -> Tensor.t -> Tensor.t
  val fit : t -> Tensor.t -> t
  val evaluate : t -> Tensor.t -> float
  val get_transforms : t -> transformation array
  val optimize_kernels : t -> Tensor.t -> t
end