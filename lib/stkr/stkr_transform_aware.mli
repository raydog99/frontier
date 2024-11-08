open Torch

type transform_aware_model = {
    base_model: Stkr_core.model;
    polynomial_degree: int;
    optimal_rate: float ref;
  }

val create_polynomial_transform : int -> Types.transform_fn
val fit : transform_aware_model -> Tensor.t -> Tensor.t -> Tensor.t -> 
    Tensor.t * Tensor.t