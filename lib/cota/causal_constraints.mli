open Torch

val truncated_factorization : 
  Scm.t -> Types.intervention -> Tensor.t -> Tensor.t
val do_calculus_distance : 
  Types.transport_plan -> Types.transport_plan -> 
  Types.intervention -> Types.intervention -> Tensor.t