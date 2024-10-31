open Torch

val compute_initial_bound : samples:Tensor.t -> epsilon:float -> Tensor.t
val verify_bound : bound:Tensor.t -> true_cov:Tensor.t -> bool
val improve_bound : current:Tensor.t -> mean_est:Tensor.t -> 
samples:Tensor.t -> epsilon:float -> Tensor.t