open Torch

type t
type encoder_type = Discrete | Continuous

val create_discrete : num_classes:int -> condition_dim:int -> t
val create_continuous : input_dim:int -> condition_dim:int -> t
val encode : t -> Tensor.t -> Tensor.t