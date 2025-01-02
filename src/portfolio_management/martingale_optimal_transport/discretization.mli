open Torch

type discretization_params = {
  theta: float;
  m_theta: float;
  dimension: int;
  truncation_radius: float;
}

type quantization_method = 
  | Uniform
  | Lloyd
  | PrincipalCurve
  | OptimalVoronoi

val create_params : theta:float -> m_theta:float -> dimension:int -> 
  truncation_radius:float -> discretization_params
val discretize : DiscreteMeasure.t -> int -> discretization_params -> 
  quantization_method -> DiscreteMeasure.t
val compute_error : DiscreteMeasure.t -> DiscreteMeasure.t -> float


module type RandomDiscretization = sig
  type sampling_method =
    | Pure
    | Stratified
    | ImportanceBased
    | QuasiMonteCarlo

  type sampling_params = {
    method_type: sampling_method;
    dimension: int;
    confidence_level: float;
    seed: int option;
  }

  val sample : DiscreteMeasure.t -> int -> sampling_params -> Tensor.t * Tensor.t
  val estimate_error : DiscreteMeasure.t -> DiscreteMeasure.t -> sampling_params -> 
    float * (float * float)
end