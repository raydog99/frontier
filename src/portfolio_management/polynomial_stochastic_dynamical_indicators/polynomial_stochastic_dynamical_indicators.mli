open Torch

(* Factorial of n *)
val factorial : int -> int

(* Binomial coefficient (n choose k) *)
val binomial : int -> int -> int

(* Multinomial coefficient *)
val multinomial : int array -> int

(* Natural logarithm *)
val ln : float -> float

(* Get the maximum value and its index from a tensor *)
val tensor_max : Tensor.t -> Tensor.t * Tensor.t

(* Compute eigenvalues of a symmetric matrix *)
val eigvals_symmetric : Tensor.t -> Tensor.t

(* Orthogonal polynomial basis functions *)
module OrthogonalPolynomials : sig
  (* Type of orthogonal polynomial basis *)
  type basis_type = Chebyshev | Legendre | Hermite

  (* Evaluate a 1D orthogonal polynomial at a point *)
  val eval_1d : basis_type -> int -> float -> float

  (* Evaluate the weight function for a basis type at a point *)
  val weight_fun : basis_type -> float -> float

  (* Compute the inner product of two basis functions *)
  val inner_product : basis_type -> int -> int -> float

  (* Get recurrence relation coefficients *)
  val recurrence_coeffs : basis_type -> int -> float * float * float

  (* Generate quadrature points and weights *)
  val quadrature_rule : basis_type -> int -> Tensor.t * Tensor.t

  (* Evaluate a multivariate basis function *)
  val eval_mv : basis_type -> int array -> float array -> float
end

(* Polynomial Chaos Expansion module *)
module PolynomialChaos : sig
  (* Polynomial chaos expansion model type *)
  type t = {
    coefs: Tensor.t;           (* Coefficients matrix [dim_state, num_terms] *)
    basis_type: OrthogonalPolynomials.basis_type;  (* Type of basis functions *)
    max_degree: int;           (* Maximum polynomial degree *)
    dim_state: int;            (* Dimension of state vector *)
    dim_params: int;           (* Dimension of parameter vector *)
    num_terms: int;            (* Total number of terms in expansion *)
  }

  (* Count the number of terms in a PCE *)
  val count_terms : int -> int -> int

  (* Create a new PCE model *)
  val create : OrthogonalPolynomials.basis_type -> int -> int -> int -> t

  (* Convert multi-index to flat index *)
  val multi_to_flat_index : int array -> int -> int

  (* Convert flat index to multi-index *)
  val flat_to_multi_index : int -> int -> int -> int array

  (* Evaluate PCE at a specific parameter point *)
  val evaluate : t -> float array -> Tensor.t

  (* Calculate PCE coefficients using non-intrusive spectral projection *)
  val compute_coefficients_nisp : 
    t -> (float -> float array -> float array -> float array) -> 
    Tensor.t -> Tensor.t -> t

  (* Calculate PCE coefficients using intrusive Galerkin method *)
  val compute_coefficients_galerkin : 
    t -> (float -> float array -> float array -> float array) -> 
    Tensor.t -> float array -> float -> int -> t

  (* Compute the mean of a PCE *)
  val mean : t -> Tensor.t

  (* Compute the variance of a PCE *)
  val variance : t -> Tensor.t

  (* Compute the covariance matrix from a PCE *)
  val covariance : t -> Tensor.t
end

(* Finite-Time Lyapunov Exponents module *)
module FTLE : sig
  (* Compute the FTLE for a deterministic system *)
  val compute : 
    (float -> float array -> float array -> float array) -> 
    Tensor.t -> float -> float -> float
end

(* Stochastic Finite-Time Lyapunov Exponents module *)
module SFTLE : sig
  (* Statistical moments of FTLE due to parameter uncertainty *)
  type sftle1 = {
    mean: float;        (* First moment (α₁¹) *)
    variance: float;    (* Second moment (α₁²) *)
    skewness: float;    (* Third moment (α₁³) *)
    kurtosis: float;    (* Fourth moment (α₁⁴) *)
  }

  (* Measure of divergence between polynomial expansions *)
  type sftle2 = {
    coefficients: float array;  (* SFTLE2 value for each coefficient (α₂ⁱ) *)
    max_value: float;           (* Maximum SFTLE2 value across coefficients *)
  }

  (* Compute SFTLE Type 1 from PCE of the FTLE *)
  val compute_type1 : 
    (float -> float array -> float array -> float array) -> 
    Tensor.t -> Tensor.t -> float -> float -> 
    OrthogonalPolynomials.basis_type -> int -> sftle1

  (* Compute SFTLE Type 2 using PCE coefficient dynamics *)
  val compute_type2 : 
    PolynomialChaos.t -> 
    (float -> float array -> float array -> float array) -> 
    Tensor.t -> float array -> float -> float -> sftle2
end

(* Pseudo-Diffusion Exponent module *)
module PseudoDiffusion : sig
  (* Pseudo-diffusion exponent type *)
  type t = {
    exponent: float;             (* The pseudo-diffusion exponent α̃ *)
    component_exponents: float array;  (* Component-wise exponents α̃ⱼ *)
  }

  (* Compute the pseudo-diffusion exponent from a PCE *)
  val compute : PolynomialChaos.t -> float -> t
end

(* Uncertain Dynamical Systems module *)
module UncertainSystem : sig
  (* Uncertain dynamical system type *)
  type t = {
    dynamic_fn: float -> float array -> float array -> float array;  (* t, p, z -> dz/dt *)
    dim_state: int;      (* Dimension of state vector *)
    dim_params: int;     (* Dimension of parameter vector *)
    param_ranges: float array array;  (* Parameter ranges [|[|min1; max1|]; ...|] *)
  }

  (* Create a new uncertain dynamical system *)
  val create : 
    (float -> float array -> float array -> float array) -> 
    int -> int -> float array array -> t

  (* Generate samples of the parameter space *)
  val sample_parameters : t -> int -> Tensor.t

  (* Propagate the system with a specific parameter vector *)
  val propagate : t -> float array -> float array -> float -> float -> float array

  (* Propagate ensemble of parameter samples *)
  val propagate_ensemble : t -> float array -> Tensor.t -> float -> float -> Tensor.t

  (* Propagate PCE for the system *)
  val propagate_pce : 
    t -> float array -> OrthogonalPolynomials.basis_type -> 
    int -> float -> float -> PolynomialChaos.t
end