open Torch
open Data_loader
open Cet

type price_volume_data = {
  price: Tensor.t;
  volume: Tensor.t;
}

type earnings_data = Tensor.t

type movement = Up | Down | Hold

module CET : sig
  type t
  val create : int -> int -> int -> int -> t
  val forward : t -> price_volume_data -> earnings_data -> Tensor.t
  val cpc_loss : t -> Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t
  val classification_loss : t -> price_volume_data -> earnings_data -> Tensor.t -> Tensor.t
  val predict : t -> price_volume_data -> earnings_data -> Tensor.t * Tensor.t
  val save : t -> filename:string -> unit
  val load : t -> filename:string -> t
end

val train : CET.t -> Data_loader.t -> Data_loader.t -> int -> float -> float -> unit