open Torch

type kernel_stats = {
  lengthscale: float;
  signal_variance: float;
  spectral_norm: float option;
}

type kernel = {
  eval: Tensor.t -> Tensor.t -> Tensor.t;
  grad: Tensor.t -> Tensor.t -> Tensor.t;
  stats: kernel_stats;
  name: string;
}

type linear_operator = {
  eval: Tensor.t -> Tensor.t;
  name: string 
}

type diff_operator = {
  order: int array;
  coefficients: Tensor.t;
}

type boundary_type = 
  | Dirichlet
  | Neumann 
  | Robin of float

type boundary_segment = {
  type_: boundary_type;
  value: Tensor.t -> Tensor.t;
  normal: Tensor.t -> Tensor.t * Tensor.t;
}

type pde_coeffs = {
  diffusion: Tensor.t;
  advection: Tensor.t;
  reaction: Tensor.t;
  forcing: Tensor.t;
}

type boundary_condition = {
  dirichlet: Tensor.t -> Tensor.t;
  neumann: Tensor.t -> Tensor.t;
}

type pde_operator = {
  interior_op: Tensor.t -> Tensor.t;
  boundary_op: Tensor.t -> Tensor.t;
}

type boundary_spec = {
  condition: boundary_condition;
  segments: boundary_segment list;
  points: Tensor.t;
}

type covariance_op = {
  forward: Tensor.t -> Tensor.t -> Tensor.t;  (* K: U* -> U *)
  adjoint: Tensor.t -> Tensor.t -> Tensor.t;  (* K*: U -> U* *)
}

type system_vars = {
  u: Tensor.t;          (* Solution *)
  z1: Tensor.t;         (* Interior values *)
  z2: Tensor.t;         (* Derivative values *)
  z3: Tensor.t;         (* Interior constraint values *)
  z_boundary: Tensor.t; (* Boundary values *)
}

val rbf_kernel : kernel_stats -> Tensor.t -> Tensor.t -> Tensor.t
val matern_kernel : kernel_stats -> Tensor.t -> Tensor.t -> Tensor.t
val periodic_kernel : lengthscale:float -> period:float -> Tensor.t -> Tensor.t -> Tensor.t

val stable_cholesky : matrix:Tensor.t -> min_eig:float -> Tensor.t * float
val woodbury_inverse : k_psi_phi:Tensor.t -> k_phi_phi:Tensor.t -> gamma:float -> Tensor.t

module RKHS : sig
  type t = {
    kernel: kernel;
    inner_product: Tensor.t -> Tensor.t -> Tensor.t;
    norm: Tensor.t -> Tensor.t;
  }

  val create : kernel:kernel -> t
end

val gradient : (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t
val partial_derivative : (Tensor.t -> Tensor.t) -> Tensor.t -> dim:int -> order:int -> Tensor.t
val laplacian : (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t

module SparseSolver : sig
  type t = {
    rkhs: RKHS.t;
    inducing_points: Tensor.t;
    sample_points: Tensor.t;
    gamma: float;
    pde: pde_operator;
  }

  val create : 
    kernel:kernel -> 
    n_inducing:int -> 
    domain:float * float * float * float ->
    gamma:float ->
    pde:pde_operator ->
    t

  val solve : t -> f:Tensor.t -> g:Tensor.t -> Tensor.t
end

module ErrorAnalysis : sig
  type eigen_decomp = {
    eigvals: Tensor.t;
    eigvecs: Tensor.t;
    condition_number: float;
  }

  val stable_eigen_decomp : Tensor.t -> eigen_decomp
  val nystrom_error_bound : 
    kernel:kernel ->
    sample_points:Tensor.t ->
    inducing_points:Tensor.t ->
    r:int ->
    delta:float ->
    float
end

module AdaptiveSampling : sig
  type sampling_criterion = 
    | PredictiveVariance
    | IntegratedVariance
    | ActiveLearning

  val select_points :
    model:'a ->  (* 'a is the model type *)
    candidates:Tensor.t ->
    n_points:int ->
    criterion:sampling_criterion ->
    Tensor.t
end

module PDEOptimizer : sig
  type t = {
    radius: float ref;
    min_radius: float;
    max_radius: float;
    eta: float;
  }

  val create :
    init_radius:float ->
    min_radius:float ->
    max_radius:float ->
    eta:float ->
    t

  val optimize :
    objective:(Tensor.t -> Tensor.t) ->
    gradient:(Tensor.t -> Tensor.t) ->
    hessian:(Tensor.t -> Tensor.t) ->
    init_x:Tensor.t ->
    optimizer:t ->
    Tensor.t
end