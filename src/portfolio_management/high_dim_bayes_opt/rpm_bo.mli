open Torch

val epsilon : float
val safe_normalize : Tensor.t -> Tensor.t
val safe_matrix_inverse : Tensor.t -> Tensor.t
val safe_sqrt : float -> float

module GaussianProcess : sig
  type t = {
    mean: Tensor.t -> float;
    kernel: Tensor.t -> Tensor.t -> float;
    training_x: Tensor.t list;
    training_y: float list;
    noise_var: float;
  }

  val create : ?mean_fn:(Tensor.t -> float) -> ?noise_var:float -> 
    (Tensor.t -> Tensor.t -> float) -> t
  val squared_exp_kernel : float -> float -> Tensor.t -> Tensor.t -> float
  val predict : t -> Tensor.t -> float * float
end

val create_random_matrix : int -> int -> Tensor.t
val project : Tensor.t -> Tensor.t -> Tensor.t
val project_to_bounds : Tensor.t -> (float * float) list -> Tensor.t
val verify_manifold_dimension : Tensor.t list -> int -> bool

module GeometryAwareManifold : sig
  type manifold_type =
    | Linear of { basis: Tensor.t }
    | Spherical of { radius: float; center: Tensor.t }
    | Mixed of { linear_basis: Tensor.t; sphere_radius: float }
    | KleinBottle of { embedding_dim: int }
    | GeneralManifold of { dim: int; projection_fn: Tensor.t -> Tensor.t }

  val create_mapping : manifold_type -> (Tensor.t -> Tensor.t)
  val verify_mapping_properties : manifold_type -> Tensor.t -> bool
end

module SemiSupervisedLearning : sig
  type validation_stats = {
    supervised_loss: float;
    consistency_loss: float;
    manifold_error: float;
    gradient_norm: float;
  }

  val manifold_consistency_check : (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t -> float -> float
  val cross_validate_gamma : ('a * float) list -> Tensor.t list -> float list -> 
    (Tensor.t -> Tensor.t) -> float
  val validate_iteration : (Tensor.t -> Tensor.t) -> (Tensor.t * float) list -> 
    Tensor.t list -> validation_stats
end

val verify_backprojection_exists : Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t -> bool
val verify_projection_completeness : Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t list -> bool
val project_back : Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t -> Tensor.t option
val verify_effective_dimension : Tensor.t list -> int -> bool
val verify_distance_preservation_bounds : Tensor.t -> Tensor.t -> Tensor.t -> 
  float -> int -> int -> bool
val verify_diffeomorphism_conditions : (Tensor.t -> Tensor.t) -> Tensor.t list -> 
  float -> bool
val verify_back_projection_convergence : (Tensor.t -> Tensor.t) -> Tensor.t -> 
  int -> bool
val search_projected_space : (Tensor.t -> Tensor.t) -> (float * float) list -> 
  int -> Tensor.t

module RPMBO : sig
  type config = {
    ambient_dim: int;
    manifold_dim: int;
    projection_dim: int;
    n_init: int;
    max_iter: int;
    exploration_weight: float;
  }

  type manifold_config = {
    manifold_type: GeometryAwareManifold.manifold_type;
    projection_dim: int;
    ambient_dim: int;
  }

  type stats = {
    convergence_stats: SemiSupervisedLearning.validation_stats array;
    diffeomorphism_verified: bool;
    distance_preserved: bool;
  }

  type t

  val create : config -> manifold_config -> t
  val optimize : t -> objective:(Tensor.t -> float) -> (Tensor.t * float) list -> Tensor.t list -> 
    (Tensor.t * float) list * stats option
end