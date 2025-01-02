open Torch
open Autoencoder
open Translator

type t

val create : int -> t
val train : t -> Tensor.t -> int -> float -> unit
val extract_patterns : t -> Tensor.t -> int -> Tensor.t