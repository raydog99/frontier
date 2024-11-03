open Torch

type mask = Tensor.t

module TimeSeries : sig
  type t = {
    data: Tensor.t;
    mask: mask;
    dimensions: int;
    sequence_length: int;
  }

  val create : data:Tensor.t -> mask:mask -> t
  val split_condition_target : t -> target_mask:mask -> 
    Tensor.t * Tensor.t * mask * mask
  val generate_random_mask : t -> missing_ratio:float -> mask
end

module DataLoader : sig
  type t

  val create : data:TimeSeries.t -> batch_size:int -> shuffle:bool -> t
  val next_batch : t -> TimeSeries.t option
  val reset : t -> unit
end