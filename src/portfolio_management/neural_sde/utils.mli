open Torch

val create_feed_forward : input_dim:int -> hidden_dim:int -> output_dim:int -> Nn.t
val tensor_to_float : Tensor.t -> float
val glorot_normal_init : Var_store.t -> shape:int list -> Tensor.t