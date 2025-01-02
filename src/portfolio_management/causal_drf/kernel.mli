open Torch

type kernel_type = 
  | Gaussian of float
  | Laplace of float 
  | InverseMultiquadric of float

type kernel_params = {
  bandwidth: float;
  degree: int option;
  scale: float;
  normalize: bool;
}

type kernel_properties = {
  bounded: bool;
  characteristic: bool;
  translation_invariant: bool;
}

val evaluate_kernel : kernel_type -> Tensor.t -> Tensor.t -> Tensor.t
val compute_kernel_matrix : sample -> kernel_type -> Tensor.t
val estimate_cme : sample -> kernel_type -> Tensor.t -> Tensor.t
val approximate_kernel : sample -> int -> float -> Tensor.t
val nystrom_approximation : sample -> int -> float -> Tensor.t
val verify_kernel_properties : kernel_type -> kernel_properties
val median_heuristic : sample -> float