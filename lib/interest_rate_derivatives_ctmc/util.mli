open Torch
open Ctmc

val check_square_matrix : Tensor.t -> unit
val check_positive_scalar : Tensor.t -> string -> unit
val check_nonnegative_tensor : Tensor.t -> string -> unit
val check_valid_time : float -> string -> unit
val check_valid_state : Ctmc.t -> int -> unit