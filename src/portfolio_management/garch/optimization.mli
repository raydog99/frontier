open Torch

val adam : 
  Tensor.t list -> 
  loss_fn:(Tensor.t list -> Tensor.t) -> 
  learning_rate:float -> 
  beta1:float -> 
  beta2:float -> 
  epsilon:float -> 
  max_iter:int -> 
  Tensor.t list