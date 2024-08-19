open Torch

type t

val create : d_x:int -> d_y:int -> device:Device.t -> t

val generate : t -> num_timesteps:int -> batch_size:int -> Tensor.t * Tensor.t