open Torch

module Manifold : sig
  type t = {
    dim: int;                    
    ambient_dim: int;            
    points: Tensor.t;            
  }

  val create : Tensor.t -> int -> int -> t
  val pairwise_distances : Tensor.t -> Tensor.t
  val geodesic_distances : Tensor.t -> int -> Tensor.t
end

module LaplaceBeltrami : sig
  type eigensystem = {
    eigenvalues: Tensor.t;
    eigenfunctions: Tensor.t;
  }

  val weight_matrix : Tensor.t -> float -> Tensor.t
  val normalized_laplacian : Tensor.t -> float -> Tensor.t
  val compute_eigensystem : Tensor.t -> int -> eigensystem
  val estimate_bounds : int -> int -> float * float
end

module HeatKernel : sig
  type t = {
    epsilon: float;
    truncation: int;
    eigenvalues: Tensor.t;
    eigenvectors: Tensor.t;
  }

  val create : float -> int -> LaplaceBeltrami.eigensystem -> t
  val evaluate : t -> Tensor.t -> Tensor.t -> float -> float
  val build_matrix : t -> Tensor.t -> float -> Tensor.t
  val local_approximation : Tensor.t -> Tensor.t -> float -> float
end