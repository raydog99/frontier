open Torch

type model_params = {
  alpha: float;
  sigma: float;
  theta: float;
  a: float;
  b: float;
  rho: float;
  t: float;
}

val create_model_params : alpha:float -> sigma:float -> theta:float -> a:float -> b:float -> rho:float -> t:float -> model_params
val create_incomplete_model_params : alpha:float -> sigma:float -> theta:float -> a:float -> b:float -> rho:float -> t:float -> model_params
val wealth_process : model_params -> Tensor.t -> (float -> Tensor.t -> Tensor.t) -> int -> Tensor.t list
val simulate_reference : model_params -> Tensor.t -> int -> Tensor.t list