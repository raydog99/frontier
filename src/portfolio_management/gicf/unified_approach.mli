open Torch
open Types

val decompose_loglikelihood : Tensor.t -> Tensor.t -> int -> 
    float * float * float * Tensor.t
    
val compute_penalties : Tensor.t -> float array -> float -> 
    {off_diagonal: float; tau: float; total: float} result
    
val regression_component : Tensor.t -> Tensor.t -> float -> 
    Tensor.t -> float -> float * float
    
val estimate : ?max_iter:int -> ?tol:float -> estimation_method -> 
    Tensor.t -> model_params -> Tensor.t result