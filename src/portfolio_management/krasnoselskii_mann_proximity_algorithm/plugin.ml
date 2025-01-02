open Torch

module type S = sig
  val name : string
  val on_iteration : int -> Tensor.t -> float -> float option -> Yojson.Safe.t
end

let load_plugin path : (module S) =
  try
    Dynlink.loadfile path
  with
  | Dynlink.Error err ->
      failwith (Printf.sprintf "Failed to load plugin %s: %s" path (Dynlink.error_message err))
  | exn ->
      failwith (Printf.sprintf "Unexpected error loading plugin %s: %s" path (Printexc.to_string exn))