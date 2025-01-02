open Torch
open Har_kernel

module type HAR = sig
  type t

  val create : ?batch_size:int -> ?early_stopping:float -> float list -> order -> t
  val fit : t -> Tensor.t -> Tensor.t -> (t, string) result
  val predict : t -> Tensor.t -> (Tensor.t, string) result
  val cross_validate : t -> Tensor.t -> Tensor.t -> int -> (t, string) result Lwt.t
  val mse : t -> Tensor.t -> Tensor.t -> (float, string) result
  val r2_score : t -> Tensor.t -> Tensor.t -> (float, string) result
  val get_params : t -> (float option * order, string) result
  val save_model : t -> string -> (unit, string) result
  val load_model : string -> (t, string) result
end

module Make (P : sig val p : int end) : HAR