open Torch

module Data : sig
  type dataset = {
    x_ts: Tensor.t;
    x_static: Tensor.t;
    y: Tensor.t;
    timestamps: float array;
  }

  exception Data_error of string

  val load_data_parallel : string -> int -> dataset

  val preprocess : dataset -> dataset

  val split_data_by_time : dataset -> float -> dataset * dataset

  val batch_data : dataset -> int -> dataset list

  val normalize : dataset -> dataset * (Tensor.t * Tensor.t)

  val denormalize : dataset -> Tensor.t * Tensor.t -> dataset

  val get_batch : dataset -> int -> int -> dataset

  val shuffle : dataset -> dataset

  val describe : dataset -> unit

  val validate_data : dataset -> unit
end