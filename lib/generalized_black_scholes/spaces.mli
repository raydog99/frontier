open Utils

module FunctionSpace : sig
  type t
  val make : float -> float -> Grid.t -> Measure.t -> t
  val v_m_norm : t -> float array -> float
  val project_v0m : t -> float array -> float array
end

module BilinearForm : sig
  type t
  val make : float -> float -> FunctionSpace.t -> t
  val evaluate : t -> float array -> float array -> Grid.t -> float
  val symmetric_part : t -> float array -> float array -> Grid.t -> float
  val nonsymmetric_part : t -> float array -> float array -> Grid.t -> float
end