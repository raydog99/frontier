open Torch

val clear : unit -> unit
  
val get_eigendecomp : 
    Tensor.t -> Tensor.t * Tensor.t
  
val get_hd_lca :
    Tensor.t -> Tensor.t -> Tensor.t