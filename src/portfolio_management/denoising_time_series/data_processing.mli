open Torch

(** Financial indicator type *)
type indicator =
  | Y10  (** 10-Year Treasury Yield *)
  | CAPE (** Cyclically Adjusted Price/Earnings Ratio *)
  | NYF  (** New York Fed Economic Activity Index *)
  | MG   (** US Corporate Margins *)
  | Y02  (** 2-Year Treasury Yield *)
  | STP  (** Steepness of the Treasury Yield Curve *)
  | M2   (** Money Supply *)

type t = {
  target: Tensor.t;
  contexts: (indicator * Tensor.t) array;
  horizon: int;
}

val indicator_of_string : string -> indicator

val string_of_indicator : indicator -> string

val load_data : Config.Config.t -> t

val prepare_input : t -> int -> int -> Tensor.t

val shift_target : t -> int -> Tensor.t