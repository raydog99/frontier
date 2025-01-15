open Torch

(** Kernel function *)
type kernel = Tensor.t -> Tensor.t -> Tensor.t

(** Density representation with log probability and gradient *)
type density = {
  log_prob: Tensor.t -> float;
  grad_log_prob: Tensor.t -> Tensor.t;
}

(** Configuration parameters for kernel computations *)
type config = {
  length_scale: float;
  nugget: float;
}

(** PDE Operators for numerical differentiation *)
module PDEOperators : sig
  (** Linear operator *)
  type linear_operator = Tensor.t -> Tensor.t

  (** Compute numerical gradient 
      @param h Step size for finite difference computation
      @param f Function to differentiate
      @param x Point of differentiation
      @return Gradient tensor
  *)
  val gradient : h:float -> (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t

  (** Compute numerical divergence 
      @param h Step size for finite difference computation
      @param f Vector field
      @param x Point of evaluation
      @return Divergence scalar
  *)
  val divergence : h:float -> (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t

  (** Compute numerical Laplacian 
      @param h Step size for finite difference computation
      @param f Function to differentiate
      @param x Point of evaluation
      @return Laplacian scalar
  *)
  val laplacian : h:float -> (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t
end

(** Vector Field operations *)
module VectorField : sig
  (** Vector field type *)
  type t = {
    apply: Tensor.t -> Tensor.t;  (** Field evaluation *)
    grad: Tensor.t -> Tensor.t;   (** Jacobian *)
    div: Tensor.t -> Tensor.t;    (** Divergence *)
  }

  (** Create a vector field 
      @param f Base function 
      @param h Step size for differentiation
      @return Vector field
  *)
  val create : f:(Tensor.t -> Tensor.t) -> h:float -> t

  (** Combine two vector fields 
      @param vf1 First vector field
      @param vf2 Second vector field
      @return Combined vector field
  *)
  val combine : t -> t -> t
end

(** Reproducing Kernel Hilbert Space (RKHS) methods *)
module RKHS : sig
  (** Compute inner product in RKHS 
      @param kernel Kernel function
      @param f First function
      @param g Second function
      @param points Evaluation points
      @return Inner product
  *)
  val inner_product : 
    kernel:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    f:(Tensor.t -> Tensor.t) -> 
    g:(Tensor.t -> Tensor.t) -> 
    points:Tensor.t -> 
    Tensor.t

  (** Compute norm in RKHS 
      @param kernel Kernel function
      @param f Function to compute norm of
      @param points Evaluation points
      @return Norm
  *)
  val norm : 
    kernel:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    f:(Tensor.t -> Tensor.t) -> 
    points:Tensor.t -> 
    Tensor.t

  (** Minimum norm interpolation 
      @param kernel Kernel function
      @param points Interpolation points
      @param values Function values
      @return Interpolating function
  *)
  val min_norm_interpolation : 
    kernel:(Tensor.t -> Tensor.t -> Tensor.t) -> 
    points:Tensor.t -> 
    values:Tensor.t -> 
    (Tensor.t -> Tensor.t)
end

(** Kernel optimization methods *)
module KernelOptimization : sig
  (** Cross-validation result *)
  type cross_validation_result = {
    optimal_length_scale: float;
    cv_scores: (float * float) array;
    validation_error: float;
  }

  (** Perform cross-validation for kernel hyperparameters 
      @param config Initial configuration
      @param density Probability density
      @param points Data points
      @param target_fn Target function
      @param k_folds Number of cross-validation folds
      @param length_scales Length scales to evaluate
      @return Cross-validation results
  *)
  val cross_validate : 
    config:config -> 
    density:density -> 
    points:Tensor.t -> 
    target_fn:(Tensor.t -> Tensor.t) -> 
    k_folds:int -> 
    length_scales:float array -> 
    cross_validation_result
end

(** Linear solver interface *)
module LinearSolver : sig
  (** Solver solution type *)
  type solution = {
    value: Tensor.t;
    iterations: int;
    error: float;
  }

  (** Conjugate gradient solver 
      @param matrix_action Linear matrix operation
      @param preconditioner Preconditioning function
      @param b Right-hand side vector
      @param tol Convergence tolerance
      @param max_iter Maximum iterations
      @return Solution
  *)
  val conjugate_gradient : 
    matrix_action:(Tensor.t -> Tensor.t) -> 
    preconditioner:(Tensor.t -> Tensor.t) -> 
    b:Tensor.t -> 
    tol:float -> 
    max_iter:int -> 
    solution

  (** MINRES solver for symmetric indefinite systems 
      @param matrix_action Linear matrix operation
      @param b Right-hand side vector
      @param tol Convergence tolerance
      @param max_iter Maximum iterations
      @return Solution
  *)
  val minres : 
    matrix_action:(Tensor.t -> Tensor.t) -> 
    b:Tensor.t -> 
    tol:float -> 
    max_iter:int -> 
    solution
end

(** Preconditioner interface *)
module PreconditionerInterface : sig
  (** Preconditioner strategy *)
  type strategy = [
    | `None
    | `Jacobi of int
    | `BlockJacobi of { block_size: int; n: int }
    | `Nystrom of {
        n_samples: int;
        nugget: float;
        sampling: [ `Uniform | `Diagonal of Tensor.t ]
      }
    | `FITC of {
        n_samples: int;
        nugget: float;
        sampling: [ `Uniform | `Diagonal of Tensor.t ]
      }
    | `RandomFeatures of {
        n_features: int;
        length_scale: float;
        nugget: float;
      }
  ]

  (** Create a preconditioner based on strategy 
      @param strategy Preconditioning strategy
      @param matrix_action Matrix operation
      @param points Data points
      @return Preconditioning function
  *)
  val create : 
    strategy -> 
    (Tensor.t -> Tensor.t) -> 
    Tensor.t -> 
    (Tensor.t -> Tensor.t)
end

(** Numerical stability utilities *)
val stable_solve : Tensor.t -> Tensor.t -> Tensor.t
val condition_number : Tensor.t -> Tensor.t
val stable_cholesky : Tensor.t -> Tensor.t

(** Kernel-related utilities *)
val langevin_operator : 
  density:density -> 
  f:(Tensor.t -> Tensor.t) -> 
  Tensor.t -> 
  Tensor.t

val stein_kernel : 
  density:density -> 
  base_kernel:(Tensor.t -> Tensor.t -> Tensor.t) -> 
  Tensor.t -> 
  Tensor.t -> 
  Tensor.t

val base_kernel : 
  length_scale:float -> 
  Tensor.t -> 
  Tensor.t -> 
  Tensor.t

(** Point sampling methods *)
val uniform : n_samples:int -> int -> int array
val diagonal : 
  n_samples:int -> 
  kernel_diag:Tensor.t -> 
  int -> 
  int array

(** Matrix operations *)
module MatrixOps : sig
  (** Matrix action type *)
  type matrix_action = {
    apply: Tensor.t -> Tensor.t;
    size: int * int;
  }

  (** Create kernel matrix action 
      @param config Kernel configuration
      @param density Probability density
      @param points Data points
      @return Matrix action
  *)
  val create_kernel_action : 
    config:config -> 
    density:density -> 
    points:Tensor.t -> 
    matrix_action

  (** Matrix-vector multiplication 
      @param action Matrix action
      @param v Vector
      @return Result of multiplication
  *)
  val mv : matrix_action -> Tensor.t -> Tensor.t
end

(** Utility functions *)
val deduplicate_points : Tensor.t -> Tensor.t
val symmetric_collocation : 
  operator:((Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t) -> 
  kernel:(Tensor.t -> Tensor.t -> Tensor.t) -> 
  points:Tensor.t -> 
  f:(Tensor.t -> Tensor.t) -> 
  Tensor.t

(** Random feature generation *)
val random_features : 
  n_features:int -> 
  length_scale:float -> 
  Tensor.t -> 
  ((Tensor.t -> Tensor.t) * Tensor.t)