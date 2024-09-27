open Torch
open Cet
open Data_loader

val train_step : CET.t -> Optimizer.t -> price_volume_data -> earnings_data -> Tensor.t -> Tensor.t -> Tensor.t -> float -> Tensor.t
val evaluate : CET.t -> Data_loader.t -> float * float

module LRScheduler : sig
  type t
  val create : Optimizer.t -> float -> float -> int -> t
  val step : t -> int -> unit
end