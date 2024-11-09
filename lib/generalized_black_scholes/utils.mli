open Torch

module type GRID = sig
  type t
  val make : float -> float -> int -> t
  val points : t -> float array 
  val delta : t -> float
  val size : t -> int
  val interior_points : t -> float array
  val is_boundary : t -> int -> bool
end

module type MEASURE = sig
  type t
  val make : ?max_depth:int -> float -> float -> float -> t
  val total_mass : t -> float
  val integrate : (float -> float) -> t -> float
  val integrate_psi : t -> int -> int -> float
  val f1 : float -> float
  val f2 : t -> float -> float
end