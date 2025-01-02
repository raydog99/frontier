open Torch

val process_large_dataset : 
  samples:Tensor.t ->
  batch_size:int ->
  f:(Tensor.t -> Tensor.t) ->
  Tensor.t

val efficient_kronecker : 
  samples:Tensor.t ->
  batch_size:int ->
  Tensor.t

val memory_efficient_multiply :
  a:Tensor.t ->
  b:Tensor.t ->
  memory_limit:int ->
  Tensor.t