open Torch
open Types

val fit : ?max_iter:int -> ?tol:float -> model_params -> Tensor.t -> 
       (Tensor.t * convergence_stats) result
       
val initialize_sigma : Tensor.t -> model_params -> Tensor.t

val compute_objective : Tensor.t -> Tensor.t -> model_params -> float

val update_sigma : Tensor.t -> int -> float -> Tensor.t -> Tensor.t