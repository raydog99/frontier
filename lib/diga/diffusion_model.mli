open Torch

type t

val create : num_timesteps:int -> channels:int -> condition_dim:int -> t
val train : t -> data:Tensor.t -> conditions:Tensor.t -> learning_rate:float -> num_epochs:int -> unit
val sample_with_guidance : t -> Tensor.t -> num_samples:int -> seq_length:int -> guidance_scale:float -> Tensor.t