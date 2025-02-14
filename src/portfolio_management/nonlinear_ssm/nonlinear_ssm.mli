open Torch

type state = Tensor.t
type observation = Tensor.t
type control = Tensor.t

type model_params = {
  transition_matrix: Tensor.t;
  observation_matrix: Tensor.t;
  process_noise: Tensor.t;
  observation_noise: Tensor.t;
}

type basis_function = {
  phi: Tensor.t -> Tensor.t;  (** Basis function evaluation *)
  grad_phi: Tensor.t -> Tensor.t;  (** Gradient of basis function *)
}

type hyperparams = {
  sf: float;  (** Signal variance *)
  l: float;   (** Length scale *)
  v: float;   (** Matérn smoothness *)
}

type model = {
  nx: int;  (** State dimension *)
  ny: int;  (** Observation dimension *)
  nu: int;  (** Control dimension *)
  params: model_params;
  basis: basis_function list;
}

val stabilize_cholesky : Tensor.t -> Tensor.t
val safe_inverse : Tensor.t -> Tensor.t
val log_det : Tensor.t -> Tensor.t
val compute_covariance : Tensor.t list -> Tensor.t -> Tensor.t
val condition_number : Tensor.t -> Tensor.t
val is_well_conditioned : Tensor.t -> float -> bool

module StateSpace : sig
  val create : int -> int -> int -> model_params -> basis_function list -> model
  val transition : model -> state -> control option -> state
  val observation : model -> state -> observation
  val sample_trajectory : model -> state -> control list -> int -> state list
end

module BasisFunctions : sig
  type basis_type =
    | Fourier
    | Gaussian
    | Polynomial
    | Wavelet of [`Haar | `Daubechies of int | `Morlet of float]

  val create_basis : basis_type -> 
    < nx: int; n_basis: int; l: float; .. > -> basis_function list
end

module GP : sig
  type kernel_type =
    | RBF
    | Matern of float  (** nu parameter *)
    | Periodic of float  (** period *)
    | SpectralMixture of int  (** number of components *)

  val create_kernel : kernel_type -> hyperparams -> (Tensor.t -> Tensor.t -> Tensor.t)
end

module Learning : sig
  type sufficient_stats = {
    phi: Tensor.t;    (** State transitions *)
    psi: Tensor.t;    (** State-observation cross terms *)
    sigma: Tensor.t;  (** State covariance *)
  }

  val compute_stats : state list -> sufficient_stats

  module PGAS : sig
    type particle = {
      state: Tensor.t;
      weight: float;
      ancestor: int;
      log_weight: float;
    }

    val run : model -> observation list -> int -> int -> state array
  end

  module ParameterUpdates : sig
    val update_transition_matrix : sufficient_stats -> Tensor.t -> Tensor.t -> Tensor.t
    val update_process_noise : sufficient_stats -> int -> int -> 
      (float * Tensor.t) -> Tensor.t
    val update_observation_matrix : sufficient_stats -> Tensor.t -> Tensor.t -> Tensor.t
  end
end

module ModelComposition : sig
  type higher_order_config = {
    order: int;
    coupling_method: [`Full | `Sparse | `Diagonal];
    delay_embedding: int option;
  }

  type composition_type =
    | Serial   (** Output of one feeds into input of next *)
    | Parallel (** Independent models combined *)
    | Feedback of {delay: int}  (** Output feeds back to input *)

  val create_higher_order_model : model -> higher_order_config -> model
  val compose : composition_type -> model list -> model
end