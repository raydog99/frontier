open Torch

type config = {
  input_dim: int;
  output_dim: int;
  hidden_dims: int list;
  activation_learnable: bool;
  use_skip_connections: bool;
  batch_norm_layers: bool;
}

type t

val create : config -> t
val forward : t -> Tensor.t -> Tensor.t
val parameters : t -> Tensor.t list
val train : t -> bool -> unit