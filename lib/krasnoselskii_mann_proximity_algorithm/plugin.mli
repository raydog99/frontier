open Torch

module type S = sig
  val name : string
  val on_iteration : int -> Tensor.t -> float -> float option -> Yojson.Safe.t
end

val load_plugin : string -> (module S)