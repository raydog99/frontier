open Torch

type t

val create :
n_ref:int -> n_sim:int -> d_y:int -> n_pos:int -> n_time:int ->
hidden_dims:int list -> n_0:int -> d_x:int -> device:Device.t -> t

val forward : t -> Tensor.t -> Tensor.t * Tensor.t

val parameters : t -> Tensor.t list

val save : t -> filename:string -> unit

val load : t -> filename:string -> unit