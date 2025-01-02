open Torch

val split_list : 'a list -> int -> 'a list * 'a list
(** Split list at given index *)

val mean_std : float list -> float * float
(** Compute mean and standard deviation *)

val list_split4 : ('a * 'b * 'c * 'd) list -> 'a list * 'b list * 'c list * 'd list
(** Split list of 4-tuples into 4 lists *)

val tensor_to_float_list : Tensor.t -> float list
(** Convert tensor to float list *)

val generate_grid : float -> float -> int -> float list
(** Generate evenly spaced grid points *)