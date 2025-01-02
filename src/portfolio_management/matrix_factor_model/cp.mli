open Torch

type t

val create : int -> int -> int -> t
val fit : ?max_iter:int -> ?tol:float -> ?n_threads:int -> ?verbose:bool -> Tensor.t -> t -> (t, string) result
val transform : Tensor.t -> t -> (Tensor.t, string) result
val reconstruction_error : Tensor.t -> t -> (float, string) result
val save : t -> string -> (unit, string) result
val load : string -> (t, string) result