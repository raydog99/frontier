open Torch

module MatrixOps : sig
  val svd : Tensor.t -> Tensor.t * Tensor.t * Tensor.t
  val matrix_multiply : Tensor.t -> Tensor.t -> Tensor.t
  val transpose : Tensor.t -> Tensor.t
  val l2_norm : Tensor.t -> Tensor.t
  val make_diagonal : Tensor.t -> Tensor.t
end

module StiefelManifold : sig
  val project_tangent : Tensor.t -> Tensor.t
  val manifold_gradient : Tensor.t -> Tensor.t -> Tensor.t
  val retract : Tensor.t -> Tensor.t
  val compute_constraint_violation : Sofar.model -> int -> float
end

module Sofar : sig
  type model = {
    u: Tensor.t;
    d: Tensor.t;
    v: Tensor.t;
    rank: int;
    x: Tensor.t;
    y: Tensor.t;
    sigma: Tensor.t;
    theta: Tensor.t;
  }

  val init_model : int -> int -> int -> model
  val estimate : Tensor.t -> Tensor.t -> int -> float -> float -> float -> model
  val estimate_with_constraints : Tensor.t -> Tensor.t -> int -> float -> float -> float -> float -> model
end

module NeymanScore : sig
  type score = {
    e_k: Tensor.t;
    m_vk: Tensor.t;
    z_kk: Tensor.t;
  }

  val modified_score_function : Sofar.model -> int -> Tensor.t -> Tensor.t -> score
  val approximation_error : score -> Tensor.t -> Tensor.t -> Tensor.t
end

module NumericalStability : sig
  type stability_params = {
    eps: float;
    max_cond_number: float;
    min_eigenval: float;
  }

  val default_params : stability_params
  val check_conditioning : Tensor.t -> stability_params -> bool
  val stable_inverse : Tensor.t -> stability_params -> Tensor.t
end

module RobustConditionChecker : sig
  type condition_result = {
    satisfied: bool;
    error_margin: float;
    confidence: float;
    message: string;
  }

  val check_sparse_eigenvalues : Tensor.t -> int -> float -> float -> 
    NumericalStability.stability_params -> condition_result
  val check_singular_separation : Tensor.t -> float -> int -> float -> 
    NumericalStability.stability_params -> condition_result
  val check_orthogonality : Tensor.t -> Tensor.t -> int -> int -> 
    NumericalStability.stability_params -> condition_result
end

module ErrorTracking : sig
  type error_bounds = {
    estimation_error: float;
    numerical_error: float;
    total_error: float;
    confidence_level: float;
  }

  val track_error : RobustConditionChecker.condition_result list -> 
    NumericalStability.stability_params -> error_bounds
end

(* Strict orthogonal latent factors *)
module SofariStrict : sig
  type model = {
    sofar: Sofar.model;
    sigma: Tensor.t;
    theta: Tensor.t;
  }

  val construct_m_strict : model -> int -> Tensor.t * Tensor.t
  val construct_w_strict : model -> int -> Tensor.t -> Tensor.t -> Tensor.t
  val infer_strict : Tensor.t -> Tensor.t -> int -> model -> Tensor.t * Tensor.t
end

(* Relaxed orthogonal latent factors *)
module SofariRelaxed : sig
  val remove_previous_layers : Tensor.t -> Tensor.t -> Sofar.model -> int -> Tensor.t
  val construct_residual_matrices : Sofar.model -> int -> Tensor.t * Tensor.t array
  val construct_m_relaxed : Sofar.model -> int -> Tensor.t -> Tensor.t * Tensor.t
  val infer_relaxed : Tensor.t -> Tensor.t -> int -> Sofar.model -> Tensor.t * Tensor.t
  val infer_k1 : Tensor.t -> Tensor.t -> Sofar.model -> Tensor.t * Tensor.t
end

module AsymptoticApproximations : sig
  type approximation_quality = {
    bias: float;
    variance_ratio: float;
    convergence_rate: float;
    sample_size_requirement: int;
  }

  val verify_asymptotic_normality : Tensor.t -> int -> int -> approximation_quality
  val compute_enhanced_delta_n : Sofar.model -> int -> int -> int -> float
end