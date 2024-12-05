open Torch

module Grid : sig
  type t
  type point = float * float

  val make : int -> int -> t
  val get : t -> int -> int -> Tensor.t
  val set : t -> int -> int -> Tensor.t -> unit
end

module CompleteMesh : sig
  type point = float * float
  type mesh_type = 
    | Fine      (** Ωhx,hy *)
    | AnisX     (** Ωhx,Hy *)
    | AnisY     (** ΩHx,hy *)
    | Coarse    (** ΩHx,Hy *)
    | Partition      (** Ωhx,hy_i,j *)

  type t

  val make_mesh : int -> int -> int -> int -> mesh_type -> t
  val check_subset : t -> t -> bool
  val check_disjoint : t -> t -> bool
  val get_partitions : int -> int -> int -> int -> point array list
end

module Config : sig
  type t = {
    nx_f: int;
    ny_f: int;
    nz_f: int;
    nx_c: int;
    ny_c: int;
    nz_c: int;
    alpha: float array;
    beta: float array;
  }

  val make : int -> int -> int -> int -> int -> int -> float array -> float array -> t
  val validate : t -> (unit, string) result
end

module Projector : sig
  type t

  val make : CompleteMesh.t -> CompleteMesh.t -> t
  val apply : t -> Tensor.t -> Tensor.t
  val transpose : t -> t
  val verify_properties : t -> bool
end

module Discretization : sig
  type t = {
    aff: Tensor.t;
    afc: Tensor.t;
    acf: Tensor.t;
    acc: Tensor.t;
  }

  val build : Config.t -> t
  val tensor_product : Tensor.t -> Tensor.t -> Tensor.t
end

module SkeletonBuilder : sig
  type t

  val make : Config.t -> t
  val build_skeleton : t -> Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
end

module PartitionSolver : sig
  type t

  val make : Config.t -> t
  val solve_partitions : t -> Tensor.t -> Tensor.t -> Tensor.t
end

module Preconditioner : sig
  type t =
    | Jacobi
    | ILU of int
    | MultiGrid of int
    | BlockJacobi of int

  val create : t -> Tensor.t -> Tensor.t
  val apply : Tensor.t -> Tensor.t -> Tensor.t
end

module ASDSM : sig
  type t

  val create : Config.t -> t
  val solve : t -> Tensor.t -> float -> int -> Tensor.t
  val compute_residual : Tensor.t -> Tensor.t -> Config.t -> Tensor.t
end