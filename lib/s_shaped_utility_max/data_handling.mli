open Torch

type dataset = {
  t: Tensor.t;
  x: Tensor.t;
  r: Tensor.t;
  v: Tensor.t;
}

type data_loader = unit -> (Tensor.t * Tensor.t * Tensor.t * Tensor.t) list

val create_dataset : int -> dataset
val split_dataset : dataset -> float -> dataset * dataset
val create_data_loader : dataset -> int -> data_loader
val get_test_data : data_loader -> Tensor.t