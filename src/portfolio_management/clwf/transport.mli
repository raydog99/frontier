open Torch

val interpolate : source:Tensor.t -> target:Tensor.t -> 
  t:float -> config:model_config -> Tensor.t

val sample_ot_maps : source:Tensor.t -> target:Tensor.t -> 
  epsilon:float -> num_iters:int -> Tensor.t * Tensor.t