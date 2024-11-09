open Utils
open Spaces

module DirichletForm : sig
  type t
  val make : BilinearForm.t -> Measure.t -> t
  val check_sector_condition : t -> float array -> float array -> Grid.t -> bool
  val check_coercivity : t -> float array -> Grid.t -> bool
end

module Generator : sig
  type t
  val make : DirichletForm.t -> t
  val apply : t -> float array -> Grid.t -> float array
  val apply_adjoint : t -> float array -> Grid.t -> float array
end