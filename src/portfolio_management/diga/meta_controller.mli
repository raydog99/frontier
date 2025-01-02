open Torch

type t

val create : num_timesteps:int -> channels:int -> num_samples:int -> seq_length:int -> condition_dim:int -> encoder_type:[`Discrete of int | `Continuous of int] -> t
val train : t -> data:Tensor.t -> conditions:Tensor.t -> learning_rate:float -> num_epochs:int -> unit
val generate_market_states : t -> Tensor.t -> guidance_scale:float -> Tensor.t