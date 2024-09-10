open Torch

val gaussian_noise : unit -> Tensor.t
val geometric_brownian_motion : float -> float -> float -> float -> float
val linear_regression : Tensor.t -> Tensor.t -> float * float
val log_binning : Tensor.t -> int -> Tensor.t * Tensor.t
val safe_division : float -> float -> float
val safe_log : float -> float
exception InvalidParameter of string
val validate_positive : string -> float -> unit
val validate_non_negative : string -> float -> unit
val classify_regime : float -> int -> Types.regime
val exponential_fit : Tensor.t -> Tensor.t -> float * float
val moving_average : float list -> int -> float list
val exponential_moving_average : float list -> float -> float list