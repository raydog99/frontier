open Torch

module LinearAlgebra : sig
  val pseudo_inverse : Tensor.t -> Tensor.t
  val matrix_sqrt : Tensor.t -> Tensor.t
  val stable_cholesky : Tensor.t -> Tensor.t
  val is_positive_definite : Tensor.t -> bool
  val nearest_positive_definite : Tensor.t -> Tensor.t

  module InnerProduct : sig
    type t = {
      compute: Tensor.t -> Tensor.t -> Tensor.t;
      metric: Tensor.t option;
    }

    val euclidean : t
    val weighted : Tensor.t -> t
    val mahalanobis : Tensor.t -> t
  end

  module Decomposition : sig
    type spectral_decomposition = {
      eigenvalues: Tensor.t;
      eigenvectors: Tensor.t;
      condition_number: float;
    }

    val compute_spectral : Tensor.t -> spectral_decomposition
    val reconstruct_from_spectral : spectral_decomposition -> Tensor.t
  end

  module Stable : sig
    val add_jitter : Tensor.t -> float -> Tensor.t
    val solve : Tensor.t -> Tensor.t -> Tensor.t
    val matrix_power : Tensor.t -> float -> Tensor.t
  end

  module Correlation : sig
    val compute_correlation_matrix : Tensor.t -> Tensor.t
    val is_valid_correlation : Tensor.t -> bool
  end
end

module Spaces : sig
  module Polish : sig
    type t = {
      topology: [`Complete | `Incomplete];
      separable: bool;
      metrisable: bool;
      basis: Tensor.t list option;
    }

    val check_separability : t -> Tensor.t list -> bool
    val verify_polish_properties : t -> Tensor.t list -> bool
    val create_with_basis : int -> t
  end

  module Tangent : sig
    type t = {
      dimension: int;
      base_point: Tensor.t;
      metric: Tensor.t;
      to_tangent: Tensor.t -> Tensor.t;
      from_tangent: Tensor.t -> Tensor.t;
    }

    val create : 
      base_point:Tensor.t -> 
      metric:Tensor.t -> 
      t

    val parallel_transport : t -> t -> Tensor.t -> Tensor.t
    val metric_at_point : t -> Tensor.t -> Tensor.t
  end

  module RiemannianManifold : sig
    type t = {
      dimension: int;
      metric: Tensor.t -> Tensor.t;
      christoffel: Tensor.t -> Tensor.t;
      exp_map: Tensor.t -> Tensor.t -> Tensor.t;
      log_map: Tensor.t -> Tensor.t -> Tensor.t;
    }

    val create : 
      dimension:int ->
      metric:(Tensor.t -> Tensor.t) ->
      christoffel:(Tensor.t -> Tensor.t) ->
      t

    val geodesic : t -> Tensor.t -> Tensor.t -> float -> Tensor.t
    val parallel_transport : 
      t -> 
      Tensor.t -> 
      Tensor.t -> 
      Tensor.t -> 
      Tensor.t
  end

  module FiberBundle : sig
    type t = {
      base_manifold: RiemannianManifold.t;
      fiber_dimension: int;
      total_space_dim: int;
      projection: Tensor.t -> Tensor.t;
      lift: Tensor.t -> Tensor.t;
      connection: Tensor.t -> Tensor.t -> Tensor.t;
    }

    val create : 
      base_manifold:RiemannianManifold.t -> 
      fiber_dim:int -> 
      t

    val horizontal_lift : t -> Tensor.t -> Tensor.t -> Tensor.t
    val fiber_transport : 
      t -> 
      Tensor.t -> 
      Tensor.t -> 
      Tensor.t -> 
      Tensor.t
    val verify_bundle_structure : t -> bool
  end

  module MetricGeometry : sig
    type metric_space = {
      dimension: int;
      distance: Tensor.t -> Tensor.t -> float;
      ball: Tensor.t -> float -> Tensor.t list;
    }

    val create_euclidean : int -> metric_space
    val verify_metric_properties : 
      metric_space -> 
      Tensor.t list -> 
      bool
  end
end

module AdjustedSpace : sig
  type t = {
    dimension: int;
    base_space: Spaces.RiemannianManifold.t;
    adjustment: Tensor.t;
    residual_space: Tensor.t -> Tensor.t;
  }

  val create : 
    base_space:Spaces.RiemannianManifold.t -> 
    adjustment:Tensor.t -> 
    t

  val inner_product : t -> Tensor.t -> Tensor.t -> Tensor.t
  val norm : t -> Tensor.t -> Tensor.t
  val project : t -> Tensor.t -> Tensor.t

  module Sequential : sig
    type adjustment_sequence = {
      spaces: t list;
      composition: t option;
    }

    val create : t list -> adjustment_sequence
    val adjust : adjustment_sequence -> Tensor.t -> Tensor.t
  end
end