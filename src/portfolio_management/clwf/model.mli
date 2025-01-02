open Torch

module MultiHeadAttention : sig
  type t
  val create : int -> int -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module TransformerBlock : sig
  type t
  val create : int -> int -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module FlowNetwork : sig
  type t
  val create : input_dim:int -> num_layers:int -> num_heads:int -> t
  val forward : t -> Tensor.t -> Tensor.t
end

module VAE : sig
  type t
  val create : input_dim:int -> latent_dim:int -> num_layers:int -> num_heads:int -> t
  val forward : t -> Tensor.t -> Tensor.t * Tensor.t * Tensor.t
  val loss : recon:Tensor.t -> input:Tensor.t -> 
    mu:Tensor.t -> logvar:Tensor.t -> Tensor.t
end