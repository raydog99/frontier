open Torch

type t

val create :
data:(Tensor.t * Tensor.t) array ->
batch_size:int ->
shuffle:bool ->
device:Device.t ->
t

val batches : t -> (Tensor.t * Tensor.t) array Seq.t