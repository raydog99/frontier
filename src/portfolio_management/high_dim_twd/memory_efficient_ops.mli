open Torch

val stream_tensor : 
  tensor:Tensor.t -> 
  f:(Tensor.t -> int -> unit) -> 
  batch_size:int -> unit
val compute_diffusion_operator : 
  Tensor.t -> Config.t -> Tensor.t