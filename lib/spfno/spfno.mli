open Torch

type boundary_condition =
  | Dirichlet
  | Neumann
  | Robin of float * float  (* a*u + b*u' = c *)

type domain = {
  dimensions: int list;
  bounds: (float * float) list;
}

(* Function spaces *)
module FunctionSpace : sig
  type norm = L2Norm | H1Norm | H1_0Norm
  val compute_norm : Tensor.t -> norm -> Tensor.t
  val numerical_derivative : Tensor.t -> Tensor.t
end

(* Transform operations *)
module Transform : sig
  val dst : Tensor.t -> Tensor.t 
  val dct : Tensor.t -> Tensor.t
  val idst : Tensor.t -> Tensor.t
  val idct : Tensor.t -> Tensor.t
  val normalize_basis_functions : Tensor.t -> domain -> Tensor.t
  val projection_filter : Tensor.t -> boundary_condition -> Tensor.t
end

(* Spectral operator *)
module SpectralOperator : sig
  type t
  
  type mode_selection = {
    max_modes: int;
    cutoff_threshold: float;
    adaptive: bool;
  }

  val create : ?mode_selection:mode_selection -> 
               width:int -> 
               bc:boundary_condition -> 
               n_dims:int -> t

  val forward : t -> Tensor.t -> Tensor.t
  val enforce_boundary_conditions : t -> Tensor.t -> Tensor.t
end

(* SPFNO module *)
module SPFNO : sig
  type t

  val create : width:int -> 
              depth:int -> 
              modes:int ->
              bc:boundary_condition -> t

  val forward : t -> Tensor.t -> Tensor.t
end

(* Error analysis *)
module ErrorAnalysis : sig
  type error_metric = 
    | L2Error
    | H1Error 
    | MaxError
    | BoundaryError

  val compute_error : predicted:Tensor.t -> 
                     target:Tensor.t -> 
                     error_metric -> float

  val analyze_stability : SPFNO.t -> Tensor.t -> Tensor.t -> float
end

(* Multi-dimensional operations *)
module HigherDim : sig
  type tensor_decomp = {
    core: Tensor.t;
    factors: Tensor.t list;
  }

  val hosvd : Tensor.t -> int -> tensor_decomp
  val tt_decomposition : Tensor.t -> float -> Tensor.t list
  val fft_nd : Tensor.t -> Tensor.t
end

(* Stability analysis *)
module MultiDimStability : sig
  type stability_measure = {
    condition_number: float;
    spectral_radius: float;
    energy_ratio: float;
    max_eigenvalue: float;
  }

  val von_neumann_analysis : Tensor.t -> float -> float -> float
  val check_cfl : Tensor.t -> float -> float -> bool
  val analyze_energy_stability : SPFNO.t -> Tensor.t -> stability_measure
end

(* Adaptive refinement *)
module AdaptiveRefinement : sig
  type refinement_criterion =
    | EnergyBased of float
    | ErrorBased of float
    | HybridCriterion of float * float

  val select_modes : Tensor.t -> refinement_criterion -> int
  val refine_mesh : Tensor.t -> Tensor.t -> Tensor.t
end