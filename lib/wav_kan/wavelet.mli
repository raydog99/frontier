open Torch

module Base : sig
  type property = {
    support_width: float;
    vanishing_moments: int;
    regularity: float;
    symmetry: [`Symmetric | `Antisymmetric | `None];
  }

  type family = 
    | Ricker of property
    | Morlet of property * float
    | Paul of property * int
    | DOG of property * int
    | Shannon of property
    | Meyer of property
    | ComplexGaussian of property * int

  val get_family : string -> family
end

module Function : sig
  val ricker : float -> float
  val morlet : float -> float -> float
  val paul : float -> int -> float
  val dog : float -> int -> float
  val meyer : float -> float
  val to_tensor_op : (float -> float) -> Tensor.t -> Tensor.t
end

module Analysis : sig
  val compute_spectrum : Tensor.t -> (float -> float) -> Tensor.t -> Tensor.t
  val extract_ridges : Tensor.t -> float -> Tensor.t
  val compute_localization : Tensor.t -> float * float
end