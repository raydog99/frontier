open Torch

type t
type point
type mixing_function = private {
  mix: float array -> point array -> point;  (* η function *)
  constant: float;                           (* C_η *)
  power: int;                               (* p *)
}

val create_mixing: float -> int -> mixing_function
val distance: point -> point -> float
val metric_capacity: float -> int                  (* θ_K(ε) *)
val quantization_modulus: float -> int            (* Q_K(ε) *)
val mix: mixing_function -> float array -> point array -> point
val quantize: int -> Tensor.t -> point
val verify_simplicial: mixing_function -> point array -> bool
val compress: point -> float -> point
val decompress: point -> point